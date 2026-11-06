"""Start only FiMA's Firefly stack, then check authenticated API readiness."""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from fima.firefly import FireflyClient, FireflyError, FireflyHTTPError, FireflyUnavailableError


COMPOSE_FILE = Path(__file__).resolve().parents[2] / "deploy/firefly/compose.yaml"


def _stop_command(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_command(command: list[str], *, cwd: Path, deadline: float, limit: float) -> tuple[int, str]:
    remaining = min(limit, deadline - time.monotonic())
    if remaining <= 0:
        raise FireflyError("Timed out starting Firefly. Check the Docker engine and repeat the command.")
    # Keep Docker diagnostics out of CLI output: they may contain interpolated
    # deployment secrets. A process group lets us clean up Docker's CLI/plugin
    # children while retaining the terminal session for cached sudo authorization.
    with tempfile.TemporaryFile() as diagnostics:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=diagnostics, process_group=0)
        except OSError as error:
            raise FireflyError("Could not launch Docker. Check the Docker installation and access permissions.") from error
        try:
            status = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise FireflyError("Timed out starting Firefly. It may still be starting; repeat the command to check readiness.") from error
        finally:
            _stop_command(process)
        diagnostics.seek(0, os.SEEK_END)
        diagnostics.seek(max(0, diagnostics.tell() - 4096))
        return status, diagnostics.read().decode("utf-8", errors="replace").casefold()


def _start_stack(compose_file: Path, deadline: float) -> None:
    if not compose_file.is_file():
        raise FireflyError("Automatic Firefly startup needs deploy/firefly/compose.yaml in the FiMA checkout. See README.md.")
    environment = compose_file.with_name(".env")
    if not environment.is_file():
        raise FireflyError("Firefly deployment secrets are missing. Run scripts/setup_firefly.py and complete the initial setup in README.md.")
    docker = shutil.which("docker")
    if docker is None:
        raise FireflyError("Docker was not found. Complete the initial Docker/Firefly setup in README.md.")
    base = [docker, "--context", "default"]

    def run(command: list[str], limit: float = 10) -> tuple[int, str]:
        return _run_command(command, cwd=compose_file.parent, deadline=deadline, limit=limit)

    if run([*base, "compose", "version"])[0] != 0:
        raise FireflyError("The Docker Compose plugin is unavailable. Install docker-compose and complete the initial setup in README.md.")
    info = ["info", "--format", "{{.ServerVersion}}"]
    status, diagnostics = run([*base, *info])
    if status != 0 and "permission denied" in diagnostics:
        sudo = shutil.which("sudo")
        if sudo is None:
            raise FireflyError("Docker socket permission denied. Configure Docker access for your user; see README.md.")
        base = [sudo, "-n", *base]
        status, diagnostics = run([*base, *info])
        if status != 0 and any(text in diagnostics for text in ("password", "sudo:", "permission denied")):
            raise FireflyError("Docker access requires authorization. Run 'sudo -v' once in this terminal, then repeat the FiMA command, "
                               "or configure Docker access for your user. Run FiMA as your normal user.")
    if status != 0:
        raise FireflyError("Docker engine is unavailable. Start it with 'sudo systemctl start docker', then repeat the FiMA command. "
                           "FiMA starts the Firefly containers, not the Docker service.")
    status, _ = run([*base, "compose", "--project-name", "fima-firefly", "--file", str(compose_file),
                     "--env-file", str(environment), "up", "--detach", "--no-recreate", "--pull", "never"], 60)
    if status != 0:
        raise FireflyError("Could not start FiMA's Firefly containers. Check Docker access, cached images, deployment configuration "
                           "and port 8088. Complete the initial setup in README.md; no images are downloaded automatically.")


def ensure_firefly(
    client: FireflyClient, cash_account_id: str, *,
    progress: Callable[[str], None] | None = None,
    timeout: float = 120, compose_file: Path = COMPOSE_FILE,
) -> dict:
    """Reuse a ready API, or bring up our managed endpoint and wait for it.

    Only connection failures can launch Compose. HTTP 5xx means wait for the
    existing service; auth/account errors and unexpected responses stop at once.
    Containers remain running after success, failure or interruption for review.
    """
    report = progress if progress is not None else lambda message: None
    report("Checking Firefly before AI processing...")
    deadline = time.monotonic() + timeout
    original_timeout = client.timeout
    started, waiting = False, False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FireflyError("Timed out waiting for Firefly's API. Check Firefly and Docker, then repeat the command; AI was not started.")
            # check() makes two GETs. Bound their combined socket timeout by
            # the remaining readiness budget, without changing POST timeouts.
            client.timeout = min(2, remaining / 2)
            try:
                result = client.check(cash_account_id)
                report("Firefly is ready; credentials and USD cash account verified.")
                return result
            except FireflyUnavailableError as error:
                if not started and not waiting:
                    if client.host not in {"127.0.0.1", "localhost"} or client.port != 8088:
                        raise FireflyError("The configured Firefly server is unavailable. Automatic startup is limited to "
                                           "FiMA's deployment at 127.0.0.1:8088; start your configured server and retry.") from error
                    report("Starting FiMA's Firefly and MariaDB containers...")
                    started = True
                    _start_stack(compose_file, deadline)
            except FireflyHTTPError as error:
                if error.status < 500:
                    raise
            if not waiting:
                report("Waiting for Firefly's API to become ready...")
                waiting = True
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    finally:
        client.timeout = original_timeout
