"""Run local audio processing and connect to Firefly III from the terminal."""

import argparse
import getpass
import json
import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from fima.extraction import ExtractionError, extract_transactions
from fima.firefly import (
    DEFAULT_DATA_DIR, DEFAULT_URL, FireflyError, check_firefly,
    configure_firefly, create_smoke_transaction,
)
from fima.processing import DEFAULT_MODEL_PATH, ProcessingError, extract_with_model, process_audio
from fima.importing import ImportingError, process_and_post
from fima.transcription import TranscriptionError, transcribe_audio


def capture_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an ISO capture timestamp, including timezone offset.") from error
    if timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("Capture timestamp must include a timezone offset.")
    return timestamp


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fima", description="Process English recordings locally and connect to Firefly III."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    transcription = commands.add_parser(
        "transcribe", help="Transcribe an audio file using large-v3 on the NVIDIA GPU."
    )
    transcription.add_argument("audio", type=Path, help="Path to a WAV, M4A, MP3, or other audio file.")
    transcription.add_argument(
        "--offline",
        action="store_true",
        help="Use cached model weights only; do not contact the model download service.",
    )
    extraction = commands.add_parser("extract", help="Propose transactions from an English transcript.")
    extraction.add_argument("transcript", help="Transcript text; use '-' to read standard input.")
    extraction.add_argument("--model-path", type=Path,
                            help=f"GGUF file for automatic server startup (default: {DEFAULT_MODEL_PATH}).")
    extraction.add_argument("--model", help="Connect to an existing server using this alias (default alias: fima-extractor).")
    extraction.add_argument("--port", type=int, help="Connect to an existing server on this port (default: 8080 with --model).")
    extraction.add_argument(
        "--captured-at", type=capture_timestamp,
        help="Original capture timestamp with offset; defaults omitted dates and resolves today/yesterday.",
    )
    processing = commands.add_parser(
        "process", help="Transcribe audio, then propose transactions using a temporary local model."
    )
    processing.add_argument("audio", type=Path, help="Path to the recording.")
    processing.add_argument(
        "--offline", action="store_true", help="Use only cached Whisper weights; extraction is always offline."
    )
    processing.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Extraction GGUF file (default: {DEFAULT_MODEL_PATH}).",
    )
    processing.add_argument(
        "--captured-at", type=capture_timestamp,
        help="Original capture timestamp with offset; defaults omitted dates and resolves today/yesterday.",
    )
    processing.add_argument("--post", action="store_true",
                            help="Save results and import complete USD entries into Firefly for review.")
    processing.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                            help="Credential and import-journal directory for --post (default: data/).")
    firefly = commands.add_parser("firefly", help="Configure and test the local Firefly III connection.")
    firefly.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="Local credential directory (default: data/; keep private).")
    firefly_commands = firefly.add_subparsers(dest="firefly_command", required=True)
    configure = firefly_commands.add_parser("configure", help="Save a personal access token after checking the cash account.")
    configure.add_argument("--url", default=DEFAULT_URL, help="Local Firefly URL (default: %(default)s).")
    configure.add_argument("--cash-account-id", required=True, help="USD asset account ID shown in its Firefly URL.")
    firefly_commands.add_parser("check", help="Check authentication and the configured USD cash account.")
    firefly_commands.add_parser("smoke-test", help="Create one fictional $0.01 expense in Firefly; delete it after checking.")
    arguments = parser.parse_args(argv)
    if arguments.command == "extract" and arguments.model_path is not None and (
        arguments.model is not None or arguments.port is not None
    ):
        parser.error("--model-path cannot be combined with existing-server options --model/--port.")

    try:
        if arguments.command == "transcribe":
            output = transcribe_audio(arguments.audio, offline=arguments.offline)
        elif arguments.command == "extract":
            transcript = sys.stdin.read() if arguments.transcript == "-" else arguments.transcript
            if arguments.model is not None or arguments.port is not None:
                proposal = extract_transactions(
                    transcript, model=arguments.model if arguments.model is not None else "fima-extractor",
                    port=arguments.port if arguments.port is not None else 8080,
                    captured_at=arguments.captured_at,
                )
            else:
                proposal = extract_with_model(
                    transcript, model_path=arguments.model_path or DEFAULT_MODEL_PATH,
                    captured_at=arguments.captured_at,
                )
            output = json.dumps(proposal, indent=2, ensure_ascii=False)
        elif arguments.command == "process":
            processor = process_and_post if arguments.post else process_audio
            posting_options = {"data_dir": arguments.data_dir} if arguments.post else {}
            result = processor(
                arguments.audio, offline=arguments.offline, model_path=arguments.model_path,
                captured_at=arguments.captured_at,
                progress=lambda message: print(f"fima: {message}", file=sys.stderr, flush=True),
                **posting_options,
            )
            output = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            if arguments.firefly_command == "configure":
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", getpass.GetPassWarning)
                        token = getpass.getpass("Firefly personal access token (hidden): ")
                except getpass.GetPassWarning as error:
                    raise FireflyError("Run configure in an interactive terminal that can hide token input.") from error
                result = configure_firefly(arguments.url, token, arguments.cash_account_id, data_dir=arguments.data_dir)
            elif arguments.firefly_command == "check":
                result = check_firefly(data_dir=arguments.data_dir)
            else:
                result = create_smoke_transaction(data_dir=arguments.data_dir)
            output = json.dumps(result, indent=2, ensure_ascii=False)
    except (TranscriptionError, ExtractionError, ProcessingError, FireflyError, ImportingError, OSError, EOFError) as error:
        print(f"fima: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"fima: {arguments.command} interrupted.", file=sys.stderr)
        if arguments.command == "firefly" and arguments.firefly_command == "smoke-test":
            print("fima: Check Firefly before repeating the test; a transaction may have been created.", file=sys.stderr)
        elif arguments.command == "process" and arguments.post:
            print("fima: Import progress is saved. Repeat the same command to reconcile any uncertain write.", file=sys.stderr)
        return 130

    print(output)
    return 0
