"""Run Whisper and a temporary llama.cpp server sequentially on the PC."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fima.extraction import ExtractionError, extract_transactions


DEFAULT_MODEL_PATH = Path.home() / "models/qwen3.8/Qwen3.8-27B-UD-Q4_K_M.gguf"


class ProcessingError(Exception):
    """An input, subprocess, or model-startup problem for the combined command."""


def _stop_process(process: subprocess.Popen, *, grace_seconds: float = 10) -> None:
    """Reap our child and terminate its process group if it is still running."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if isinstance(error, KeyboardInterrupt):
            raise


def _transcribe_in_subprocess(audio_path: Path, *, offline: bool) -> str:
    command = [sys.executable, "-m", "fima", "transcribe", str(audio_path)]
    if offline:
        command.append("--offline")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", start_new_session=True,
    )
    try:
        output, errors = process.communicate()
    finally:
        _stop_process(process)
    if process.returncode == 130:
        raise KeyboardInterrupt
    if process.returncode != 0:
        detail = errors.strip().removeprefix("fima: ") or f"exit status {process.returncode}"
        raise ProcessingError(f"Transcription failed: {detail}")
    transcript = output.strip()
    if not transcript:
        raise ProcessingError("Transcription returned no text; extraction was not started.")
    return transcript


def _local_json(port: int, path: str) -> dict | None:
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            return None
        raw = response.read(65_537)
        if len(raw) > 65_536:
            return None
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except (OSError, HTTPException, ValueError):
        return None
    finally:
        connection.close()


def _startup_error(message: str, log: BinaryIO) -> ProcessingError:
    log.seek(0, os.SEEK_END)
    log.seek(max(0, log.tell() - 8192))
    detail = log.read().decode("utf-8", errors="replace").strip()
    return ProcessingError(f"{message}\n{detail}" if detail else message)


def _wait_for_server(
    process: subprocess.Popen, port: int, alias: str, log: BinaryIO,
    *, timeout: float = 180,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise _startup_error("The temporary llama.cpp server exited during startup.", log)
        if _local_json(port, "/health") is not None:
            models = _local_json(port, "/v1/models")
            if models is not None:
                entries = models.get("data", [])
                if not isinstance(entries, list) or not any(
                    isinstance(entry, dict) and entry.get("id") == alias for entry in entries
                ):
                    raise ProcessingError("The local server did not expose this run's model alias.")
                if process.poll() is None:
                    return
        time.sleep(0.2)
    raise _startup_error("Timed out waiting for the temporary llama.cpp server to load.", log)


def process_audio(
    audio_path: str | Path, *, offline: bool = False,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    captured_at: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Return a transcript and proposals requiring review, leaving audio untouched.

    Whisper runs in a child that exits before llama.cpp starts. The temporary
    server uses only local weights and is stopped on success, failure or Ctrl+C.
    Logs use an unlinked temporary file; no transcript or result file is saved.
    """
    report = progress if progress is not None else lambda message: None
    audio = Path(audio_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    try:
        for path, label in ((audio, "Audio"), (model, "Extraction model")):
            if not path.is_file() or path.stat().st_size == 0:
                raise ProcessingError(f"{label} must be a nonempty file: {path}")
    except OSError as error:
        raise ProcessingError(f"Cannot access an input file: {error}") from error
    if captured_at is not None and captured_at.utcoffset() is None:
        raise ProcessingError("Capture timestamp must include a timezone offset.")
    executable = shutil.which("llama")
    if executable:
        server_command = [executable, "serve"]
    else:
        executable = shutil.which("llama-server")
        if not executable:
            raise ProcessingError("llama.cpp was not found. Install llama or llama-server; see README.md.")
        server_command = [executable]

    report("Transcribing audio with Whisper large-v3...")
    try:
        transcript = _transcribe_in_subprocess(audio, offline=offline)
    except OSError as error:
        raise ProcessingError(f"Could not start transcription: {error}") from error
    report("Transcription complete; Whisper has released the GPU.")

    # Choose a free loopback port. Verify a unique model alias before sending any
    # transcript, so a port race cannot route extraction to an unrelated server.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    alias = f"fima-process-{uuid4().hex}"
    server_command.extend([
        "-m", str(model), "--alias", alias, "--host", "127.0.0.1", "--port", str(port),
        "--device", "CUDA0", "-ngl", "all", "-c", "4096", "-np", "1",
        "--cache-ram", "0", "--reasoning", "off", "--no-mmproj", "--no-webui", "--offline",
    ])
    report("Loading the local extraction model...")
    with tempfile.TemporaryFile() as log:
        try:
            server = subprocess.Popen(
                server_command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True,
            )
        except OSError as error:
            raise ProcessingError(f"Could not start llama.cpp: {error}") from error
        try:
            _wait_for_server(server, port, alias, log)
            report("Extracting and validating proposed transactions...")
            try:
                proposal = extract_transactions(
                    transcript, model=alias, port=port, captured_at=captured_at,
                )
            except ExtractionError as error:
                raise ProcessingError(f"Extraction failed after transcription: {error}") from error
        finally:
            _stop_process(server)
    report("Extraction model stopped; proposals are ready for review.")
    return {"transcript": transcript, **proposal}
