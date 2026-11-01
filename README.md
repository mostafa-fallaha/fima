# FiMA

A local, voice-first personal finance project, built one component at a time.
The first component turns an English audio file into a transcript on your PC.
The eventual flow is Watch → phone queue → PC → review → Firefly III.
English and USD are the initial scope; this component only transcribes audio.

## 1. Set up the Python environment

Run commands from this project directory. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
if needed, then:

```bash
uv python install 3.12
uv sync
```

`uv` creates `.venv/` for this project, without changing the system Python.
Python 3.12 is selected by `.python-version`; dependencies live in `pyproject.toml`.
The environment includes faster-whisper and NVIDIA's CUDA 12 cuBLAS/cuDNN 9
runtime packages. No PyTorch or separate FFmpeg installation is needed.

## 2. Check GPU access

This component targets Linux and the RTX 3090, using CUDA device 0 and FP16.

```bash
nvidia-smi
```

This must show your GPU before attempting real transcription. If it fails inside
a sandbox or container, check device access there as well as on the host.
The application reports a GPU error rather than silently switching to the CPU.
It does not install or change your system's NVIDIA driver.

Make the environment's NVIDIA runtime libraries visible to the dynamic loader.
Run the command for your shell in each new terminal session, from the project
directory.

For Bash or Zsh:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For Fish:

```fish
set -gx LD_LIBRARY_PATH \
    "$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib" \
    "$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
    $LD_LIBRARY_PATH
```

Fish exports path variables as colon-separated lists; `set -gx` sets and exports
the variable. Bash's `${VAR:+...}` syntax is not supported by Fish.

This follows [faster-whisper's Linux runtime setup](https://github.com/SYSTRAN/faster-whisper#gpu).
The NVIDIA driver, CUDA runtime libraries, and the Python package are different
pieces: `nvidia-smi` checks driver access, while the export above locates runtime
libraries used during inference.

## 3. Transcribe a recording

Place a recording in `recordings/` or use a path anywhere on your PC:

```bash
uv run fima transcribe /path/to/recording.m4a
```

WAV, M4A, MP3, and other formats supported by PyAV can be decoded. The recording
is read without modification. The first speech-containing recording downloads
the large-v3 model weights from Hugging Face into `.cache/models/`.
The model is several GB; subsequent runs reuse that cache. Audio is not uploaded.
No financial fields are extracted yet, so spoken amounts appear as recognized
by Whisper and still need checking by listening to the recording.

The transcript goes to standard output. Problems go to standard error and return
exit status 1; invalid arguments return 2 and an interruption returns 130.
Silence produces a no-speech message instead of a successful empty transcript.
Speech detection reduces hallucinations, but it cannot guarantee accuracy for
all noise, accents, or recordings. English is the configured recognition language,
not an automatic language check.

To save a transcript yourself:

```bash
uv run fima transcribe recordings/lunch.m4a > /tmp/lunch-transcript.txt
```

To use only previously downloaded model files:

```bash
uv run --offline fima transcribe recordings/lunch.m4a --offline
```

The first `--offline` belongs to uv and disables package downloads. The second
belongs to FiMA and disables model-download requests. Complete setup and an
initial transcription first, then repeat this command with networking disabled.
Use the same project directory so the relative model cache can be found.

## 4. Understand the code

- `src/fima/cli.py` parses the command and prints the result or a readable error.
- `src/fima/transcription.py` contains the reusable `transcribe_audio()` function.
  It checks the file, decodes audio to mono 16 kHz, rejects audio without detected
  speech, checks the GPU, and runs large-v3 in English transcription mode.
- `src/fima/__main__.py` also allows `python -m fima`.
- `tests/` exercises error handling and pipeline behavior with runtime stand-ins.
  These tests do not measure recognition quality or prove that CUDA works.

The function consumes every generated segment before returning the complete
transcript. If inference fails midway, a partial transcript is not printed as
though it were complete. The later PC backend can call the same function.

## 5. Validate this component

Run the automated checks without a model or GPU:

```bash
uv run python -m unittest discover -s tests -v
```

During environment setup, the same checks can run using an available Python
3.12+ interpreter without installing inference dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For real recognition validation, record and listen back to a few short samples:

| Say | Check in the transcript |
| --- | --- |
| I bought chicken for ten dollars. | ten dollars |
| I paid fifteen dollars and fifty cents for lunch yesterday. | 15.50 and yesterday, with equivalent wording accepted |
| I received one hundred dollars from Sam. | received, 100, and Sam |
| I spent fifteen dollars, then another fifty dollars. | both amounts remain distinct |
| A few seconds of silence | no-speech error and no transcript |

Compare amounts manually; punctuation and digit-versus-word formatting can vary.
Repeat one real recording offline using the cached weights. Complete these GPU
and audio checks before treating the component as validated for real recordings.

Recordings, model weights, caches, and environments are excluded from Git.
Keep saved transcripts outside the repository or inside the ignored
`recordings/` directory. Transcripts contain personal information too.

## Current validation status

The standard-library tests passed using the environment's Python 3.14. The
target Python 3.12 environment has not yet been installed or validated here.

The implementation environment could not resolve download hosts, so Python 3.12,
inference dependencies, and a dependency lock could not yet be downloaded or
resolved there. It also exposed no NVIDIA device files. Live transcription,
real silence detection, and the offline model-cache check therefore remain
pending on a GPU-accessible PC after setup.

Once network access is available, generate and retain the full dependency lock:

```bash
uv lock
uv sync --locked
```

Commit the resulting `uv.lock` alongside `pyproject.toml` after successful
resolution. Subsequent setup should use `uv sync --locked` to reproduce those
dependency versions. The currently pinned faster-whisper version and PyAV bound
avoid the newer PyAV API change that removed `metadata_errors`; the full
transitive dependency lock still needs generation.
