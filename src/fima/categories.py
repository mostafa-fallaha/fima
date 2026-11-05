"""Set up the initial categories and discover the user's Firefly categories."""

from pathlib import Path

from fima.firefly import DEFAULT_DATA_DIR, FireflyClient, FireflyError, load_firefly


# Firefly categories are flat; these labels are recorded in new category notes.
CATEGORY_GROUPS = {
    "Food & Drinks": ("Groceries", "Restaurants", "Snacks & Drinks"),
    "Transportation": ("Fuel", "Car Maintenance", "Public Transportation"),
    "Shopping": ("Clothing", "Electronics", "Personal Items"),
    "Entertainment": ("Gaming", "Outings & Activities", "Subscriptions"),
    "Health & Fitness": ("Healthcare", "Gym & Fitness"),
    "Housing": ("Rent", "Utilities", "Home Supplies"),
    "Finance": ("Fees & Charges", "Investments"),
    "Income": ("Salary", "Other Income"),
    "Other": ("Gifts", "Miscellaneous", "Others"),
}

FALLBACK_CATEGORY = "Others"
CATEGORY_NAMES = tuple(name for names in CATEGORY_GROUPS.values() for name in names)
_CATEGORY_LOOKUP = {name.casefold(): name for name in CATEGORY_NAMES}


def normalize_category(value: str | None) -> str:
    """Use only the configured names; unassigned/unknown labels become Others."""
    return _CATEGORY_LOOKUP.get(value.strip().casefold(), FALLBACK_CATEGORY) if value else FALLBACK_CATEGORY


def _category(data: dict) -> dict:
    try:
        identifier, name = data["id"], data["attributes"]["name"]
        if (not isinstance(identifier, str) or not identifier.isascii()
                or not identifier.isdigit() or int(identifier) < 1
                or not isinstance(name, str) or not name.strip()):
            raise ValueError
        return {"id": identifier, "name": name}
    except (KeyError, TypeError, ValueError) as error:
        raise FireflyError("Firefly returned incomplete category information.") from error


def _list_categories(client: FireflyClient) -> dict[str, dict]:
    """Read every page before creating anything; don't follow remote URLs."""
    categories = {}
    page, total_pages = 1, None
    while True:
        response = client.request("GET", f"/categories?limit=100&page={page}")
        try:
            records = response["data"]
            pagination = response["meta"]["pagination"]
            pages = pagination["total_pages"]
            if (not isinstance(records, list) or type(pages) is not int
                    or not 1 <= pages <= 1000 or type(pagination["current_page"]) is not int
                    or pagination["current_page"] != page
                    or (total_pages is not None and pages != total_pages)):
                raise ValueError
            total_pages = pages
            for record in records:
                category = _category(record)
                key = category["name"].strip().casefold()
                if key in categories:
                    raise ValueError
                categories[key] = category
            if page == total_pages:
                return categories
            page += 1
        except (KeyError, TypeError, ValueError) as error:
            raise FireflyError("Firefly returned incomplete or conflicting category pages. No categories were created.") from error


def list_categories(client: FireflyClient) -> list[dict]:
    """Discover the user's actual categories, including custom names."""
    categories = list(_list_categories(client).values())
    if len({category["id"] for category in categories}) != len(categories):
        raise FireflyError("Firefly returned conflicting category IDs.")
    return categories


def setup_categories(*, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Create only missing names; never update or delete existing categories.

    POSTs aren't retried. A fresh invocation lists categories again, including
    successful writes whose replies were lost. No import journal or AI is used.
    """
    client, _ = load_firefly(data_dir)
    categories = _list_categories(client)
    created, existing = [], []
    for group, names in CATEGORY_GROUPS.items():
        for name in names:
            key = name.casefold()
            if key in categories:
                existing.append({**categories[key], "group": group})
                continue
            try:
                response = client.request("POST", "/categories", {"name": name, "notes": "Group: " + group})
                category = _category(response["data"])
                if category["name"] != name:
                    raise FireflyError("Firefly returned a different category name.")
            except (FireflyError, KeyError, TypeError) as error:
                detail = str(error) if isinstance(error, FireflyError) else "Invalid category response."
                raise FireflyError(
                    f"Could not confirm category '{name}'. Earlier categories may already have been created. "
                    "Repeat setup-categories to check existing names before continuing. " + detail
                ) from error
            categories[key] = category
            created.append({**category, "group": group})
    return {"created_count": len(created), "existing_count": len(existing),
            "created": created, "existing": existing}
