# FiMA

A local, voice-first personal finance project, built one component at a time.
The first component turns an English audio file into a transcript on your PC.
The second proposes transactions from that transcript through local llama.cpp.
`fima process` connects both stages in one command and manages their GPU lifetimes.
The eventual flow is Watch → phone queue → PC → Firefly III → review there.
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

FiMA automatically locates and loads the cuBLAS/cuDNN libraries installed by
`uv sync` in the active Python environment. Both `transcribe` and `process` work
without setting `LD_LIBRARY_PATH`, in Fish, Bash, or Zsh. No shell configuration
change is needed. If those runtime packages are missing or damaged, run
`uv sync --locked` again.

`nvidia-smi` checks access to the system driver; FiMA loads the separate runtime
libraries that Whisper needs for inference.

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
- `src/fima/cuda.py` loads the environment's Linux cuBLAS/cuDNN libraries by
  absolute path before the GPU check, without relying on shell exports.
- `src/fima/extraction.py` calls a local llama.cpp server, validates proposed
  amounts and source quotes, and resolves supported date phrases in Python.
- `src/fima/processing.py` manages temporary llama.cpp servers for text/audio
  extraction and evaluations, and runs transcription in a separate child process.
- `src/fima/firefly.py` configures a local Firefly API connection and creates
  one fictional test transaction. Recording imports are a later step.
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

FiMA automatically starts the downloaded Gemma model, extracts the result, and
stops its server on success, failure, or Ctrl+C. A second terminal is unnecessary:

```bash
uv run --offline fima extract "I bought chicken for 10 dollars."
```

The default GGUF is `~/models/gemma_4_e2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf`.
To select another local model, pass `--model-path /path/to/model.gguf`.

The JSON result contains an expense for `"10.00"` USD, a description, its exact
source quote, `date: null`, and a `missing_date` review issue. All successful
results include `requires_review: true`. These are proposals; `extract` and
`process` do not import recordings into Firefly yet. Section 8 tests the connection
with a separate fictional transaction.

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
errors. If you explicitly want to use an already-running server, pass `--port`
(and optionally `--model` for its alias). For example, `--port 8080` connects to
the existing `fima-extractor` alias without starting or stopping a server.
Existing-server options cannot be combined with `--model-path`.

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

After setting up the Python environment and checking GPU access in sections 1–2:

```bash
uv run --offline fima process /path/to/recording.mp3 --offline
```

The command uses your existing model at
`~/models/gemma_4_e2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf`. It checks that the recording,
model file and `llama` (or `llama-server`) executable are available before
starting inference. It never installs a model or changes the original recording.

Whisper transcribes in a separate Python process. That process exits before
the extraction model starts, releasing its GPU memory. FiMA then starts a
temporary llama.cpp server on a free port at `127.0.0.1`, waits until its own
model is ready, and sends the transcript for extraction. The server uses local
GGUF weights, CUDA0, thinking disabled, and a 4096-token context.

Both `process` and `extract` manage their own servers by default. If you have an
older manually started server running, you can stop it in its terminal to free
GPU memory. FiMA stops only the child processes it creates.
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

To test another local GGUF, such as the previous Qwen model:

```bash
uv run --offline fima process /path/to/recording.mp3 --offline \
    --model-path /home/mostafa/models/qwen3.8/Qwen3.8-27B-UD-Q4_K_M.gguf
```

The `--offline` after `process` controls Whisper's cached weights. The temporary
llama.cpp server always runs offline; the first `--offline` belongs to uv.
The standalone `transcribe` and `extract` commands remain available for checking
each stage separately. All financial proposals still require manual review.

## 8. Set up Firefly III and test the connection

Firefly III will be the finance and review UI. The agreed workflow is to import
complete, validated entries with `needs-review`, correct or delete them in
Firefly, and remove that tag once reviewed. The tag is our review convention;
tagged entries already affect balances. Incomplete or ambiguous recording results
will stay local when recording imports are implemented.

This first connector only configures access, checks a USD asset account, and
creates a clearly fictional $0.01 connection test. It uses the standard library
and does not change the recording-processing commands.

The included Compose setup follows the [official Firefly Docker setup](https://github.com/firefly-iii/docker),
with Firefly 6.7.9, MariaDB LTS, persistent database/upload volumes, and the web
port bound to `127.0.0.1:8088`. It uses English and Asia/Beirut time. Scheduled
jobs can be added when recurring transactions are needed.

This Arch setup uses the system Docker engine and Compose plugin. If needed,
install the plugin and start the engine in your terminal:

```bash
sudo pacman -S docker-compose
sudo systemctl start docker
```

Then, from the FiMA directory:

```bash
uv run --offline python scripts/setup_firefly.py
sudo docker --context default compose -f deploy/firefly/compose.yaml up -d
```

The explicit `default` context uses the system Docker engine, avoiding this PC's
inactive Docker Desktop context. The setup script creates ignored
`deploy/firefly/.env` secrets with mode 0600 and preserves an existing file.
Keep those secrets with database backups. The first Docker image pull requires
internet access and can take time. After migrations finish, open
[Firefly III](http://localhost:8088).

1. Register your own local Firefly login and choose USD as the primary currency.
2. Create a USD asset account named **Cash USD**. Choose any opening balance
   yourself; FiMA does not invent or set one. Open the account and note its
   numeric ID in the URL.
3. In **Options → Profile → OAuth → Personal Access Tokens**, create a token
   named **FiMA**. Paste it into the hidden terminal prompt below, replacing
   `ACCOUNT_ID` with the account's numeric ID:

```bash
uv run --offline fima firefly configure --cash-account-id ACCOUNT_ID
uv run --offline fima firefly check
uv run --offline fima firefly smoke-test
```

The token is saved in ignored `data/firefly.json` with mode 0600, so you do not
need to export it in each terminal. It is never printed. Requests go directly to
loopback, without HTTP proxies or redirects. A failed configuration check
preserves the previous configuration.

`smoke-test` creates **one real ledger entry** for $0.01, labelled
**FiMA connection test — delete after verifying**, with `needs-review` and
`fima-smoke-test` tags. Its JSON output links to the transaction. Inspect its
amount, source cash account, notes and editable fields, then delete it in
Firefly. Each successful invocation creates a new test; it is not an import or
retry command. It leaves the test expense account available for future checks.
If a POST fails or is interrupted, check Firefly before repeating it: the write
may already have succeeded. The connector never automatically retries a write.

To inspect startup or stop the service while keeping its data:

```bash
sudo docker --context default compose -f deploy/firefly/compose.yaml logs --tail=50 app
sudo docker --context default compose -f deploy/firefly/compose.yaml stop
```

The user has now started Firefly and configured FiMA. On 2026-11-05, a read-only
API check successfully authenticated to Firefly 6.7.9 and confirmed the configured
active USD asset account. The test transaction's creation/viewing has not been
independently verified; the check did not create another transaction.

## Current validation status

As of 2026-11-05, Python 3.12.13, the inference dependencies, `uv.lock`, and the
complete large-v3 weights are present. All 116 standard-library tests pass using
the project interpreter. These include extraction validation, client error
handling, CLI behavior, automatic CUDA library loading, evaluation scoring,
sequential processing, and cleanup.
Firefly tests cover configuration permissions, loopback/authentication checks,
USD asset-account validation, request mapping, review tags, redacted errors,
and write failures without retries. These use API stand-ins, not a running
Firefly installation. Docker image tags and Compose YAML syntax were checked;
the user subsequently started Firefly, and its authenticated read-only API check
passed on 2026-11-05. A real write smoke test has not been independently verified.
Cleanup tests include real child processes that exit on SIGTERM or require
SIGKILL after the grace period. The user previously confirmed real transcription
using a Fish library-path export; automatic loading now replaces that workaround.
On 2026-11-03, both `uv run --offline fima transcribe ... --offline` and
`uv run --offline fima process ... --offline` succeeded on the existing recording
with `LD_LIBRARY_PATH` unset before starting each process. The recording was
unchanged, and no FiMA inference children remained afterward.

Systematic recognition-quality checks and real-silence evaluation remain to be
completed. The automated tests use runtime stand-ins, so their success does not
establish recognition accuracy. Earlier network/GPU restrictions in the agent's
sandbox should not be confused with the user's working host setup.

Extraction was also exercised on the RTX 3090 using the already-present
Qwen3.8-27B-UD-Q4_K_M model. It passed 19 of 20 fabricated development cases
at roughly 2.2 seconds per example. The remaining shared-date/two-purchase case
produced an invented source quote and was rejected by validation. Clarifying
the quote instruction did not change that result. The cents/yesterday CLI
smoke test succeeded. The originally planned Qwen3.5-9B/Gemma Q8_0 comparison
remains incomplete; those partial downloads were stopped.

On 2026-11-05 the user's downloaded **Gemma 4 E2B IT UD-Q8_K_XL** was tested
using an automatically managed server. It passed **15/20**, with 18 structurally
valid proposals and a mean of **0.595 s** per case versus Qwen's earlier
**19/20**, **2.175 s**. Cases, prompt/schema hashes, and extraction settings match.
The downloaded 5,282,807,904-byte file is a different quantization from the
originally planned Q8_0 candidate. These are development checks, not accuracy
guarantees or a completed quantization comparison.

Gemma failed shared-date quote validation and correction handling; it also
misclassified ambiguous received money, a hypothetical purchase, and transcript
instructions as transactions. The latter interpretation errors passed structural
validation. Gemma is the user's trial default, with these limits preserved.

Automatic `extract` passed a known amount/date check, and automatic `process`
returned a transcript and one proposal from the existing recording without
modifying it, with `LD_LIBRARY_PATH` unset. Interrupting another Gemma text run
returned 130, produced no stdout, and removed its server. No Firefly writes were
made during these tests. Descriptions and financial meaning still require review.

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
