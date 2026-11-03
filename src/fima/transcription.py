"""Decode audio, check for speech, and transcribe it locally with Whisper."""

from pathlib import Path

from .cuda import load_cuda_libraries


class TranscriptionError(Exception):
    """An audio, setup, or inference problem that can be shown to the user."""


def transcribe_audio(audio_path: str | Path, *, offline: bool = False) -> str:
    """Return an English transcript, leaving the input recording unchanged.

    CUDA device 0 and FP16 are intentional defaults for the RTX 3090. Model
    weights are cached under .cache/models in the current working directory.
    Nothing is written to the recording's directory.
    """
    path = Path(audio_path).expanduser()
    try:
        if not path.is_file():
            raise TranscriptionError(f"Audio file does not exist or is not a file: {path}")
        if path.stat().st_size == 0:
            raise TranscriptionError(f"Audio file is empty: {path}")
    except OSError as error:
        raise TranscriptionError(f"Cannot access audio file: {error}") from error

    # Lazy imports let help and input validation work without loading AI libraries.
    try:
        import av
        import ctranslate2
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.utils import download_model
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except (ImportError, OSError) as error:
        raise TranscriptionError(
            f"Transcription dependencies could not load. Run 'uv sync' and follow "
            f"the GPU setup in README.md. Details: {error}"
        ) from error

    try:
        audio = decode_audio(str(path), sampling_rate=16000)
    except (OSError, ValueError, EOFError, IndexError, av.error.FFmpegError) as error:
        raise TranscriptionError(f"Could not decode audio file '{path}': {error}") from error

    if audio.size == 0:
        raise TranscriptionError("No speech detected: the file contains no decoded audio.")

    # This inexpensive PC-side check avoids downloading/loading Whisper for silence.
    try:
        speech = get_speech_timestamps(audio, VadOptions())
    except Exception as error:
        raise TranscriptionError(f"Speech detection failed: {error}") from error
    if not speech:
        raise TranscriptionError("No speech detected in the recording.")

    try:
        load_cuda_libraries()
        if ctranslate2.get_cuda_device_count() < 1:
            raise TranscriptionError(
                "No NVIDIA GPU is accessible. Check 'nvidia-smi' on the PC and "
                "GPU access in this execution environment. CPU fallback is disabled."
            )
        if "float16" not in ctranslate2.get_supported_compute_types("cuda", 0):
            raise TranscriptionError("The accessible NVIDIA GPU does not support FP16 inference.")
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        raise TranscriptionError(
            f"CUDA initialization failed. Run 'uv sync' to restore GPU runtime libraries; "
            f"check 'nvidia-smi' and the GPU setup in README.md. "
            f"Details: {error}"
        ) from error

    try:
        cache_directory = Path.cwd() / ".cache" / "models"
        model_directory = Path(
            download_model(
                "large-v3", cache_dir=str(cache_directory), local_files_only=offline
            )
        )
        # WhisperModel otherwise attempts a remote tokenizer fallback, even when
        # local_files_only=True. Require it before entering that constructor.
        if not (model_directory / "tokenizer.json").is_file():
            raise ValueError(
                "The cached model is incomplete: tokenizer.json is missing. "
                "Restore the tokenizer or download the complete model while online."
            )
        model = WhisperModel(
            str(model_directory),
            device="cuda",
            device_index=0,
            compute_type="float16",
            local_files_only=offline,
        )
    except Exception as error:
        cache_hint = (
            "Offline mode requires weights already cached in this project."
            if offline
            else "The first run needs internet access to download model weights."
        )
        raise TranscriptionError(
            f"Could not load large-v3. {cache_hint} Check CUDA 12/cuDNN 9 setup "
            f"and available GPU memory. Details: {error}"
        ) from error

    try:
        segments, _ = model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        # Inference is lazy: fully consume segments before returning/printing text.
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    except Exception as error:
        raise TranscriptionError(
            f"Transcription failed. Check GPU memory and CUDA 12/cuDNN 9 setup. Details: {error}"
        ) from error

    if not transcript:
        raise TranscriptionError("No speech could be transcribed from the recording.")
    return transcript
