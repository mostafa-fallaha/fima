"""Connect to local Firefly III through its authenticated JSON API."""

import json
import os
import tempfile
from datetime import datetime
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo


class FireflyError(Exception):
    """A Firefly connection or configuration problem suitable for CLI output."""


DEFAULT_URL = "http://127.0.0.1:8088"
DEFAULT_DATA_DIR = Path("data")


class FireflyClient:
    def __init__(self, url: str, token: str):
        if not isinstance(url, str):
            raise FireflyError("Firefly URL must be a local HTTP URL.")
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and not parsed.username and not parsed.password
                and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
            )
            port = parsed.port if parsed.port is not None else 80
            valid = valid and 1 <= port <= 65535
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise FireflyError("Firefly URL must be an HTTP loopback address without a path, e.g. " + DEFAULT_URL)
        if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
            raise FireflyError("A Firefly personal access token is required (without whitespace).")
        self.url = url.rstrip("/")
        self.host, self.port, self.token = parsed.hostname, port, token

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        connection = HTTPConnection(self.host, self.port, timeout=20)
        headers = {"Accept": "application/json", "Authorization": "Bearer " + self.token}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, "/api/v1" + path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status not in {200, 201}:
                advice = "Check the personal access token." if response.status in {401, 403} else "Check the Firefly server and account setup."
                raise FireflyError(f"Firefly returned HTTP {response.status}. {advice}")
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise FireflyError("Firefly response exceeded 1 MiB.")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Expected an object")
            return result
        except (OSError, HTTPException) as error:
            # Do not echo request details or the token, and never retry a POST.
            raise FireflyError("Cannot complete the local Firefly request. Check that Firefly is running.") from error
        except (ValueError, UnicodeError) as error:
            raise FireflyError("Firefly returned an invalid JSON response.") from error
        finally:
            connection.close()

    def check(self, cash_account_id: str) -> dict:
        if not isinstance(cash_account_id, str) or not cash_account_id.isascii() or not cash_account_id.isdigit() or int(cash_account_id) < 1:
            raise FireflyError("Cash account ID must be a positive integer from Firefly's account URL.")
        try:
            about = self.request("GET", "/about")["data"]
            account = self.request("GET", "/accounts/" + cash_account_id)["data"]
            attributes = account["attributes"]
            if attributes["type"] != "asset" or attributes["currency_code"] != "USD":
                raise FireflyError("Choose a USD asset account for cash; FiMA will not convert currencies.")
            if attributes.get("active") is False:
                raise FireflyError("The selected cash account is inactive.")
            return {"version": about["version"], "cash_account_id": cash_account_id,
                    "cash_account_name": attributes["name"], "currency": "USD"}
        except (KeyError, TypeError) as error:
            raise FireflyError("Firefly returned incomplete account/version information.") from error

    def find_external_id(self, external_id: str) -> str | None:
        """Find exactly one split with this reference, including across pages.

        A missing result is NOT proof that an earlier timed-out POST failed.
        Never follow pagination URLs supplied by the remote server.
        """
        matches = []
        page = 1
        while True:
            query = urlencode({"query": 'external_id_is:"' + external_id + '"',
                               "limit": 100, "page": page})
            response = self.request("GET", "/search/transactions?" + query)
            try:
                groups = response["data"]
                pagination = response["meta"]["pagination"]
                pages = pagination["total_pages"]
                if (not isinstance(groups, list) or type(pages) is not int
                        or not 1 <= pages <= 1000 or pagination["current_page"] != page):
                    raise ValueError
                for group in groups:
                    identifier = transaction_id({"data": group})
                    splits = group["attributes"]["transactions"]
                    if not isinstance(splits, list) or not splits:
                        raise ValueError
                    found = [split for split in splits if split["external_id"] == external_id]
                    # An unsupported/ignored search operator must fail closed.
                    if not found:
                        raise ValueError
                    matches.extend(identifier for _ in found)
                if page >= pages:
                    break
                page += 1
            except (KeyError, TypeError, ValueError) as error:
                raise FireflyError("Firefly returned incomplete external-ID search results; no write was attempted.") from error
        if len(matches) > 1:
            raise FireflyError("Several Firefly entries share the import external ID. Resolve them in Firefly before continuing.")
        return matches[0] if matches else None


def transaction_id(response: dict) -> str:
    """Validate the group ID returned by a create or search request."""
    try:
        identifier = response["data"]["id"]
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdigit() or int(identifier) < 1:
            raise ValueError
        return identifier
    except (KeyError, TypeError, ValueError) as error:
        raise FireflyError("Firefly returned an invalid transaction ID.") from error


def configure_firefly(url: str, token: str, cash_account_id: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    client = FireflyClient(url, token)
    result = client.check(cash_account_id)
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".firefly-", dir=data_dir)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump({"url": client.url, "token": token, "cash_account_id": cash_account_id}, output)
            output.write("\n")
        os.replace(temporary, data_dir / "firefly.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def load_firefly(data_dir: Path) -> tuple[FireflyClient, str]:
    try:
        config = json.loads((data_dir / "firefly.json").read_text())
        return FireflyClient(config["url"], config["token"]), config["cash_account_id"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FireflyError("Run 'fima firefly configure --cash-account-id ID' first.") from error


def check_firefly(*, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    client, cash_account_id = load_firefly(data_dir)
    return client.check(cash_account_id)


def create_smoke_transaction(*, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Create one real ledger entry clearly marked as a fictional $0.01 test."""
    client, cash_account_id = load_firefly(data_dir)
    client.check(cash_account_id)
    payload = {
        "apply_rules": False,
        "fire_webhooks": False,
        "transactions": [{
            "type": "withdrawal",
            "date": datetime.now(ZoneInfo("Asia/Beirut")).date().isoformat(),
            "amount": "0.01",
            "currency_code": "USD",
            "description": "FiMA connection test — delete after verifying",
            "source_id": cash_account_id,
            "destination_name": "FiMA test expenses",
            "tags": ["needs-review", "fima-smoke-test"],
            "notes": "Fictional connection test, not an actual purchase. Delete this entry after checking it.",
            "external_id": "fima-smoke-" + uuid4().hex,
            "reconciled": False,
        }],
    }
    try:
        response = client.request("POST", "/transactions", payload)
        identifier = response["data"]["id"]
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdigit():
            raise ValueError("Invalid transaction ID")
    except (FireflyError, KeyError, TypeError, ValueError) as error:
        raise FireflyError(
            "Could not confirm the test transaction. Check Firefly before running smoke-test again; "
            "the request may have succeeded. " + (str(error) if isinstance(error, FireflyError) else "Invalid transaction response.")
        ) from error
    return {"transaction_id": identifier, "url": client.url + "/transactions/show/" + identifier,
            "amount": "0.01", "currency": "USD", "requires_review": True,
            "delete_after_verifying": True}
