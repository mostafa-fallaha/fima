"""Translate command-line arguments into a transcription request."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fima.transcription import TranscriptionError, transcribe_audio


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fima", description="Transcribe English recordings locally on your PC."
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
    arguments = parser.parse_args(argv)

    try:
        transcript = transcribe_audio(arguments.audio, offline=arguments.offline)
    except TranscriptionError as error:
        print(f"fima: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("fima: transcription interrupted.", file=sys.stderr)
        return 130

    print(transcript)
    return 0
