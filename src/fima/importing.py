"""Import recording proposals with durable progress, through Firefly's API."""

import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fima.extraction import ExtractionError, TRANSACTION_SCHEMA, validate_proposal
from fima.firefly import DEFAULT_DATA_DIR, FireflyError, load_firefly, transaction_id
from fima.processing import DEFAULT_MODEL_PATH, process_audio


class ImportingError(Exception):
    """A journal or import problem suitable for CLI output."""


def _audio_hash(audio: Path) -> str:
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ImportingError("Audio must be a nonempty regular file.")
    with audio.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@contextmanager
def _journal(data_dir: Path) -> Iterator[sqlite3.Connection]:
    """Serialize local importers, including inference and network requests.

    SQLite commits use FULL synchronization before each remote write. The file
    lock is separate so we never hold a SQLite transaction across a POST.
    """
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_fd = os.open(data_dir / "imports.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ImportingError("Another FiMA importer is using this data directory. Wait for it to finish.") from error
        path = data_dir / "imports.sqlite3"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS recordings (
                    audio_sha256 TEXT PRIMARY KEY,
                    captured_at TEXT,
                    model_path TEXT NOT NULL,
                    result TEXT NOT NULL,
                    firefly_url TEXT,
                    cash_account_id TEXT
                );
                CREATE TABLE IF NOT EXISTS entries (
                    audio_sha256 TEXT NOT NULL REFERENCES recordings(audio_sha256),
                    ordinal INTEGER NOT NULL,
                    external_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'uncertain', 'posted')),
                    firefly_id TEXT,
                    PRIMARY KEY (audio_sha256, ordinal)
                );
            """)
            yield connection
        finally:
            connection.close()
    finally:
        os.close(lock_fd)


def _validate_result(
    result: dict, captured_at: datetime | None, *, allow_date_default: bool = False,
) -> dict:
    """Revalidate saved/generated proposals at the ledger boundary."""
    try:
        transcript = result["transcript"]
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError
        transactions = []
        for item in result["transactions"]:
            if set(item) != set(TRANSACTION_SCHEMA["required"]) | {"date"}:
                raise ValueError
            transactions.append({key: item[key] for key in TRANSACTION_SCHEMA["required"]})
        checked = {"transcript": transcript, **validate_proposal(
            {"transactions": transactions, "issues": result["issues"]},
            transcript, captured_at=captured_at,
        )}
        if checked != result:
            if not allow_date_default or captured_at is None:
                raise ValueError
            # Upgrade only the omitted dates from the old policy. All source
            # fields, other dates and non-date issues must still match exactly.
            upgraded = {**result, "transactions": [
                {**item, "date": captured_at.date().isoformat()}
                if item["date_text"] is None and item["date"] is None else item
                for item in result["transactions"]
            ], "issues": [issue for issue in result["issues"] if issue != "missing_date"]}
            if checked != upgraded:
                raise ValueError
        return checked
    except (ExtractionError, KeyError, TypeError, ValueError) as error:
        raise ImportingError("Invalid saved or generated proposal; no entries were sent to Firefly.") from error


def _prepare_entries(db: sqlite3.Connection, digest: str, result: dict) -> None:
    if result["transactions"] and not result["issues"]:
        db.executemany("INSERT OR IGNORE INTO entries VALUES (?, ?, ?, 'pending', NULL)", [
            (digest, index, f"fima-{digest}-{index + 1}")
            for index in range(len(result["transactions"]))
        ])


def _payload(entry: dict, result: dict, cash_account_id: str, external_id: str) -> dict:
    transaction = {
        "type": "withdrawal" if entry["type"] == "expense" else "deposit",
        "amount": entry["amount"], "currency_code": "USD", "date": entry["date"],
        "description": entry["description"], "external_id": external_id,
        "tags": ["needs-review"], "reconciled": False,
        "notes": "Imported by FiMA; review against the original recording.\n"
                 "External ID: " + external_id + "\n\nOriginal transcript:\n" + result["transcript"],
    }
    if entry["type"] == "expense":
        transaction.update(source_id=cash_account_id, destination_name="FiMA expenses")
    else:
        transaction.update(source_name="FiMA income", destination_id=cash_account_id)
    return {"apply_rules": False, "fire_webhooks": False, "transactions": [transaction]}


def process_and_post(
    audio_path: str | Path, *, offline: bool = False,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    captured_at: datetime | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Save proposals before importing; resume without rerunning AI or edits.

    Content hashes identify duplicate files, not similar spoken purchases.
    Uncertain POSTs are only adopted when found; an absent search result never
    triggers another POST because the original request may still be committing.
    """
    report = progress if progress is not None else lambda message: None
    try:
        audio = Path(audio_path).expanduser().resolve()
        directory = Path(data_dir).expanduser().resolve()
        if captured_at is not None and captured_at.utcoffset() is None:
            raise ImportingError("Capture timestamp must include a timezone offset.")
        timestamp = captured_at.isoformat() if captured_at is not None else None
        with _journal(directory) as db:
            digest = _audio_hash(audio)
            saved = db.execute("SELECT * FROM recordings WHERE audio_sha256 = ?", (digest,)).fetchone()
            attempted = db.execute(
                "SELECT 1 FROM entries WHERE audio_sha256 = ? AND status != 'pending'", (digest,),
            ).fetchone()
            if saved is not None and saved["captured_at"] != timestamp:
                if attempted:
                    raise ImportingError("This recording already has import attempts with a different capture timestamp. "
                                         "Repeat the original --captured-at value; correct posted entries in Firefly.")
                # Held/unattempted recordings may be resolved with capture metadata.
                saved = None
            if saved is None:
                result = _validate_result(process_audio(
                    audio, offline=offline, model_path=model_path,
                    captured_at=captured_at, progress=report,
                ), captured_at)
                if _audio_hash(audio) != digest:
                    raise ImportingError("The audio changed during processing; no entries were sent to Firefly.")
                with db:
                    db.execute("DELETE FROM entries WHERE audio_sha256 = ?", (digest,))
                    db.execute("DELETE FROM recordings WHERE audio_sha256 = ?", (digest,))
                    db.execute("INSERT INTO recordings(audio_sha256, captured_at, model_path, result) VALUES (?, ?, ?, ?)",
                               (digest, timestamp, str(Path(model_path).expanduser().resolve()), json.dumps(result)))
                    _prepare_entries(db, digest, result)
            else:
                report("Using the saved proposal and import progress; AI will not run again.")
                original = json.loads(saved["result"])
                result = _validate_result(original, captured_at, allow_date_default=not attempted)
                if result != original:
                    with db:
                        db.execute("UPDATE recordings SET result = ? WHERE audio_sha256 = ?",
                                   (json.dumps(result), digest))
                        _prepare_entries(db, digest, result)
                    report("Applied the capture-date default to saved, unattempted entries.")

            def summary(status: str, url: str | None = None) -> dict:
                entries = [dict(row) for row in db.execute(
                    "SELECT ordinal + 1 AS entry, external_id, status, firefly_id FROM entries "
                    "WHERE audio_sha256 = ? ORDER BY ordinal", (digest,),
                )]
                for entry in entries:
                    if entry["firefly_id"] and url:
                        entry["url"] = url + "/transactions/show/" + entry["firefly_id"]
                return {**result, "import": {"recording_id": digest, "status": status,
                        "journal": str(directory / "imports.sqlite3"), "entries": entries}}

            if not result["transactions"] or result["issues"]:
                report("Recording held locally: " + ", ".join(result["issues"]) + ". No Firefly writes.")
                return summary("held")

            saved = db.execute("SELECT * FROM recordings WHERE audio_sha256 = ?", (digest,)).fetchone()
            entries = db.execute("SELECT * FROM entries WHERE audio_sha256 = ? ORDER BY ordinal", (digest,)).fetchall()
            if [entry["ordinal"] for entry in entries] != list(range(len(result["transactions"]))):
                raise ImportingError("The import journal has incomplete entry progress. Preserve it; no new writes were attempted.")
            if all(entry["status"] == "posted" for entry in entries):
                return summary("posted", saved["firefly_url"])

            client, cash_account_id = load_firefly(directory)
            if saved["firefly_url"] is not None and (saved["firefly_url"], saved["cash_account_id"]) != (client.url, cash_account_id):
                raise ImportingError("The saved Firefly destination differs from the current configuration. "
                                     "Restore the original URL/account before resuming this recording.")
            client.check(cash_account_id)
            with db:
                db.execute("UPDATE recordings SET firefly_url = ?, cash_account_id = ? WHERE audio_sha256 = ?",
                           (client.url, cash_account_id, digest))

            for entry in entries:
                if entry["status"] == "posted":
                    continue
                external_id, ordinal = entry["external_id"], entry["ordinal"]
                identifier = client.find_external_id(external_id)
                if identifier is None:
                    if entry["status"] == "uncertain":
                        raise ImportingError("Cannot confirm the earlier POST for " + external_id + ". "
                                             "Inspect Firefly before any manual recovery; automatic reposting is disabled. "
                                             "Progress is saved in " + str(directory / "imports.sqlite3"))
                    payload = _payload(result["transactions"][ordinal], result, cash_account_id, external_id)
                    # Commit BEFORE calling the server. Ctrl+C, process death, or
                    # a local commit failure after POST all leave a recoverable marker.
                    with db:
                        db.execute("UPDATE entries SET status = 'uncertain' WHERE audio_sha256 = ? AND ordinal = ?",
                                   (digest, ordinal))
                    report(f"Posting transaction {ordinal + 1}/{len(entries)} to Firefly...")
                    try:
                        identifier = transaction_id(client.request("POST", "/transactions", payload))
                    except FireflyError as error:
                        raise ImportingError("Firefly did not confirm " + external_id + ". "
                                             "The request may have succeeded. Repeat the same command to reconcile; "
                                             "FiMA will not blindly repost. " + str(error)) from error
                with db:
                    db.execute("UPDATE entries SET status = 'posted', firefly_id = ? WHERE audio_sha256 = ? AND ordinal = ?",
                               (identifier, digest, ordinal))
            return summary("posted", client.url)
    except sqlite3.Error as error:
        raise ImportingError("Cannot read or commit the import journal. Preserve it and repeat the same command "
                             "to reconcile any interrupted write; do not delete the journal.") from error
    except (ValueError, UnicodeError) as error:
        raise ImportingError("Cannot read the saved import result. Preserve the journal; no automatic recovery was attempted.") from error
