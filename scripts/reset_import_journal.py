"""Clear FiMA's local import history for testing, preserving its tables."""

import argparse
import sqlite3
import sys
from pathlib import Path

from fima.firefly import DEFAULT_DATA_DIR
from fima.importing import ImportingError, _journal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Credential/journal directory (default: data/ under the current directory).")
    args = parser.parse_args(argv)
    directory = args.data_dir.expanduser().resolve()
    path = directory / "imports.sqlite3"
    try:
        if not path.exists():
            print(f"No import journal at {path}; nothing to reset.")
            return 0
        with _journal(directory) as db, db:
            entries = db.execute("DELETE FROM entries").rowcount
            recordings = db.execute("DELETE FROM recordings").rowcount
    except (ImportingError, sqlite3.Error, OSError) as error:
        print(f"Cannot reset the import journal: {error}", file=sys.stderr)
        return 1
    print(f"Cleared {recordings} recording(s) and {entries} import entry/entries from {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
