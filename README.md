# FiMA

A local, voice-first personal finance project, built one component at a time.
The first component turns an English audio file into a transcript on your PC.
The second proposes transactions from that transcript through local llama.cpp.
`fima process` connects both stages in one command and manages their GPU lifetimes.
The eventual flow is Watch → phone queue → PC → review → Firefly III.
English and USD are the initial scope. Extraction always requires human review.

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
- `src/fima/extraction.py` calls a local llama.cpp server, validates proposed
  amounts and source quotes, and resolves supported date phrases in Python.
- `src/fima/processing.py` runs transcription in a child process, then starts
  and stops a temporary llama.cpp server for extraction.
- `evals/` contains 20 fabricated transcripts and a model-comparison runner.
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

## 6. Propose transactions from text

This component uses the existing `llama` installation separately from Whisper.
FiMA sends text only to `127.0.0.1`, without HTTP proxies or redirects. Model
downloads are a separate setup step; `fima extract` never downloads weights.
It uses the Python standard library, with no new package dependencies.

Use an existing GGUF model, or download a candidate using
[the extraction evaluation guide](evals/README.md). Start the local server in
another terminal (replace `/path/to/model.gguf` with the real file):

```bash
llama serve -m /path/to/model.gguf \
    --alias fima-extractor --host 127.0.0.1 --port 8080 \
    --device CUDA0 -ngl all -c 4096 -np 1 --cache-ram 0 \
    --reasoning off --no-mmproj --no-webui --offline
```

Once the server reports that the model is ready:

```bash
uv run --offline fima extract "I bought chicken for 10 dollars."
```

The JSON result contains an expense for `"10.00"` USD, a description, its exact
source quote, `date: null`, and a `missing_date` review issue. All successful
results include `requires_review: true`. These are proposals; this project has
no ledger connection yet.

For relative dates, supply the **original recording's capture timestamp** with
its timezone offset, even if processing happens days later:

```bash
uv run --offline fima extract "I paid 15.50 dollars for lunch yesterday." \
    --captured-at 2026-11-02T00:30:00+02:00
```

Python resolves that date to `2026-11-01`. Supported date expressions are `today`,
`yesterday`, and explicit `YYYY-MM-DD` dates. Other expressions remain quoted
with `date: null` and `unresolved_date`. Omitting a date never silently selects
the processing date. Without a capture timestamp, relative dates stay unresolved.

You can save a transcript using the existing transcription command, then read it:

```bash
uv run --offline fima extract - < /tmp/lunch-transcript.txt
```

The standalone commands help distinguish transcription errors from interpretation
errors. Use `--port` if your local llama.cpp server uses a
different port, or `--model` if its alias differs from `fima-extractor`.

Amounts are decimal strings checked with `Decimal`, never binary floating point.
Missing amounts stay null. Explicit foreign currencies stay visible and are
flagged as unsupported, without conversion. Transfers and ambiguous incoming
money are flagged. Correction utterances create no new transactions in this
initial version. Accounts, categories, posting, and correction application are
future components.

Validation checks structure and some financial constraints; it cannot establish
that the model understood the transcript correctly. Check the source quote,
amount, type, currency and description before accepting any proposal.

## 7. Process a recording in one command

With the Python environment and CUDA library path from sections 1–2 configured:

```bash
uv run --offline fima process /path/to/recording.mp3 --offline
```

The command uses your existing model at
`~/models/qwen3.8/Qwen3.8-27B-UD-Q4_K_M.gguf`. It checks that the recording,
model file and `llama` (or `llama-server`) executable are available before
starting inference. It never installs a model or changes the original recording.

Whisper transcribes in a separate Python process. That process exits before
the extraction model starts, releasing its GPU memory. FiMA then starts a
temporary llama.cpp server on a free port at `127.0.0.1`, waits until its own
model is ready, and sends the transcript for extraction. The server uses local
GGUF weights, CUDA0, thinking disabled, and a 4096-token context.

You do not need to start a server manually for `process`. If you started the
standalone server from section 6, stop it first to free GPU memory. FiMA stops
only the child processes it creates; it does not stop other applications.
Each run loads the extraction model again and stops it after success, failure,
or Ctrl+C. Startup is limited to 180 seconds; inference requests to 120 seconds.

Stage messages go to stderr. On success, stdout contains one complete JSON
object with `transcript`, `transactions`, `issues`, and `requires_review: true`.
On failure or interruption, it prints no partial successful JSON and cleans up
its children. Exit codes remain 0, 1, 2 and 130 as described above.
Temporary server logs are discarded, and FiMA does not save transcripts/results.

Supply capture time explicitly when the recording says today or yesterday:

```bash
uv run --offline fima process /path/to/recording.mp3 --offline \
    --captured-at 2026-11-02T00:30:00+02:00
```

Use the recording's real capture timestamp and timezone offset. File modification
time and processing time are not used as substitutes. Omitting capture metadata
leaves relative dates unresolved for review.

When you download a smaller model, select its local GGUF file:

```bash
uv run --offline fima process /path/to/recording.mp3 --offline \
    --model-path /path/to/smaller-model.gguf
```

The `--offline` after `process` controls Whisper's cached weights. The temporary
llama.cpp server always runs offline; the first `--offline` belongs to uv.
The standalone `transcribe` and `extract` commands remain available for checking
each stage separately. All financial proposals still require manual review.

## Current validation status

As of 2026-11-02, Python 3.12.13, the inference dependencies, `uv.lock`, and the
complete large-v3 weights are present. All 75 standard-library tests pass using
the project interpreter. These include extraction validation, client error
handling, CLI behavior, evaluation scoring, sequential processing, and cleanup.
Cleanup tests include real child processes that exit on SIGTERM or require
SIGKILL after the grace period. The user confirmed a real recording transcribed
successfully with cached weights after configuring the Fish GPU library path.

Systematic recognition-quality checks and real-silence evaluation remain to be
completed. The automated tests use runtime stand-ins, so their success does not
establish recognition accuracy. Earlier network/GPU restrictions in the agent's
sandbox should not be confused with the user's working host setup.

Extraction was also exercised on the RTX 3090 using the already-present
Qwen3.8-27B-UD-Q4_K_M model. It passed 19 of 20 fabricated development cases
at roughly 2.2 seconds per example. The remaining shared-date/two-purchase case
produced an invented source quote and was rejected by validation. Clarifying
the quote instruction did not change that result. The cents/yesterday CLI
smoke test succeeded. The Qwen3.5-9B/Gemma E2B comparison is still pending:
model downloads were too slow, and partial transfers were stopped. These are
limited development results; descriptions still need manual review.

The combined `process` command also succeeded on the user's existing recording
with cached weights: it returned a transcript, one proposed transaction, and a
missing-date flag. The audio was unchanged, and the temporary inference
processes exited with GPU memory released. Private contents were not copied
into these notes. This smoke test verifies execution, not financial accuracy.
Interrupting another real run during Qwen startup returned 130, printed no
partial JSON, and removed the owned server while leaving the audio unchanged.

Use `uv sync --locked` to reproduce dependencies. The pinned faster-whisper
version and PyAV bound avoid the newer PyAV API change removing
`metadata_errors`. See [plan.md](plan.md) for troubleshooting history, remaining
issues, and the next component.
