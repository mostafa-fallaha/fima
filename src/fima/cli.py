"""Run local transcription and transaction extraction from the terminal."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from fima.extraction import ExtractionError, extract_transactions
from fima.processing import DEFAULT_MODEL_PATH, ProcessingError, process_audio
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
        prog="fima", description="Transcribe English recordings and propose transactions locally."
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
    extraction.add_argument("--model", default="fima-extractor", help="Local server model alias.")
    extraction.add_argument("--port", type=int, default=8080, help="Local llama.cpp port (default: 8080).")
    extraction.add_argument(
        "--captured-at", type=capture_timestamp,
        help="Original capture timestamp with offset; resolves today/yesterday (never processing time).",
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
        help="Extraction GGUF file (default: ~/models/qwen3.8/Qwen3.8-27B-UD-Q4_K_M.gguf).",
    )
    processing.add_argument(
        "--captured-at", type=capture_timestamp,
        help="Original capture timestamp with offset; resolves today/yesterday.",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "transcribe":
            output = transcribe_audio(arguments.audio, offline=arguments.offline)
        elif arguments.command == "extract":
            transcript = sys.stdin.read() if arguments.transcript == "-" else arguments.transcript
            proposal = extract_transactions(
                transcript, model=arguments.model, port=arguments.port,
                captured_at=arguments.captured_at,
            )
            output = json.dumps(proposal, indent=2, ensure_ascii=False)
        else:
            result = process_audio(
                arguments.audio, offline=arguments.offline, model_path=arguments.model_path,
                captured_at=arguments.captured_at,
                progress=lambda message: print(f"fima: {message}", file=sys.stderr, flush=True),
            )
            output = json.dumps(result, indent=2, ensure_ascii=False)
    except (TranscriptionError, ExtractionError, ProcessingError, OSError) as error:
        print(f"fima: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"fima: {arguments.command} interrupted.", file=sys.stderr)
        return 130

    print(output)
    return 0
