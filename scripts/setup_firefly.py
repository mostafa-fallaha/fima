"""Create Firefly's local Docker secrets once, without displaying them."""

import os
import secrets
import string
from pathlib import Path


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "deploy/firefly/.env"
    alphabet = string.ascii_letters + string.digits
    values = {
        name: "".join(secrets.choice(alphabet) for _ in range(32))
        for name in ("APP_KEY", "DB_PASSWORD", "DB_ROOT_PASSWORD")
    }
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print("Firefly secrets already exist; preserved the existing file.")
        return
    with os.fdopen(descriptor, "w") as output:
        output.write("".join(f"{name}={value}\n" for name, value in values.items()))
    print("Created deploy/firefly/.env with private permissions. Keep this file with your backups.")


if __name__ == "__main__":
    main()
