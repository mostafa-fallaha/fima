# FiMA

A local, voice-first personal finance project, built one component at a time.
The first component turns an English audio file into a transcript on your PC.
The eventual flow is Watch → phone queue → PC → review → Firefly III.
English and USD are the initial scope; this component only transcribes audio.

For project decisions, implementation status, and continuing in a new session,
read [plan.md](plan.md). [AGENTS.md](AGENTS.md) directs coding agents to that handoff.

## 1. Set up the Python environment

Run commands from this project directory. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
if needed, then:

```bash
uv python install 3.12
uv sync --locked
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

As of 2026-11-02, Python 3.12.13, the inference dependencies, `uv.lock`, and the
complete large-v3 weights are present. All 23 standard-library tests pass using
the project interpreter. The user confirmed a real recording transcribed
successfully with cached weights after configuring the Fish GPU library path.

Systematic recognition-quality checks and real-silence evaluation remain to be
completed. The automated tests use runtime stand-ins, so their success does not
establish recognition accuracy. Earlier network/GPU restrictions in the agent's
sandbox should not be confused with the user's working host setup.

Use `uv sync --locked` to reproduce dependencies. The pinned faster-whisper
version and PyAV bound avoid the newer PyAV API change removing
`metadata_errors`. See [plan.md](plan.md) for troubleshooting history, remaining
issues, and the next component.
