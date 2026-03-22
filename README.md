# Armenian Voice AI Bank Agent

Production-minded Armenian bank support assistant focused on three allowed topics only:

- `credits`
- `deposits`
- `branches`

The repository includes the full path from scraping official bank pages to grounded Armenian text QA and voice interaction.

## Deliverables

- GitHub-ready project repository
- Documentation with architecture, design decisions, setup, and run instructions

## What This Project Does

The system answers only from scraped official bank website data and only for:

- credits
- deposits
- branch locations

It refuses:

- out-of-scope requests
- comparative or recommendation requests
- unsupported questions
- weak-evidence questions

The current stack supports:

- scraping and ingestion
- chunking and indexing
- retrieval
- grounded QA in Armenian
- voice input/output
- self-hosted LiveKit integration
- simple local chat/voice UI

## Architecture & Decisions

### High-Level Flow

```text
Official bank pages
  -> scraping / ingestion
  -> cleaned documents
  -> chunking
  -> local retrieval index
  -> query scope guard
  -> retrieval + evidence gate
  -> grounded Armenian answer / clarification / refusal
  -> optional voice STT / TTS
  -> LiveKit or local UI
```

### Architecture

```text
app/
  config/       environment + YAML config loading
  ingestion/    scraping, parsing, normalization, storage
  chunking/     boilerplate cleanup + chunk creation
  retrieval/    index building + retrieval
  guardrails/   scope restriction
  qa/           evidence checks + grounded QA pipeline
  prompts/      inspectable grounded prompt
  voice/        STT/TTS providers + LiveKit wrapper
  ui/           simple local Gradio interface
scripts/        runnable entry points
data/           raw, processed, chunks, indexes
tests/          focused unit tests
```

### Design Decisions

#### 1. Config-driven scraping

Bank-specific URLs and rules live in YAML instead of inside the generic scraper code.

Why:

- easy to add more banks later
- safer than unrestricted crawling
- easier to explain in an interview setting

#### 2. Strict-document ingestion instead of blind crawling

For this take-home, explicit bank URLs are more reliable than broad crawling.

Why:

- better correctness
- easier debugging
- lower risk of pulling irrelevant content

#### 3. Lightweight hybrid retrieval

The retrieval layer uses a local hybrid TF-IDF approach with word and character features.

Why:

- simple and reproducible
- works reasonably well for Armenian
- easy to inspect and rebuild locally
- metadata filtering by bank/topic is straightforward

#### 4. Guardrails before generation

The QA system does not send every query directly to the LLM. It first applies:

- topic scope restriction
- retrieval
- evidence quality checks
- clarification logic for underspecified cases

Why:

- prevents unsupported answers
- keeps the assistant grounded
- makes refusal behavior explicit and testable

#### 5. Gemini for grounded QA

I chose `gemini-2.5-flash` for grounded answer generation.

Why:

- free tier availability
- fast enough for interactive use
- good Armenian understanding in practice
- better cost/practicality tradeoff than paid options for this project

I tried and compared multiple practical options during development. The final choice favored models that were both free and gave the best usable Armenian results in this setup.

#### 6. STT choice

The current speech-to-text default is local `faster-whisper` with:

- `VOICE_STT_MODEL=large-v3-turbo`

Why:

- local and simple to run
- better Armenian transcription quality than smaller tested variants
- no external STT billing dependency

#### 7. TTS choice

The current text-to-speech provider is Gemini API TTS:

- `gemini-2.5-flash-preview-tts`

Why:

- free option available
- easy integration via `GEMINI_API_KEY`
- no Google Cloud ADC / service-account setup required

Important note:

- official Gemini API TTS docs do not currently list `hy-AM` as a supported `languageCode`
- this project still steers Armenian generation through prompt instructions, and keeps `VOICE_TTS_LANGUAGE=hy-AM` in config for transparency

#### 8. LiveKit as a thin voice transport layer

LiveKit is not used for intelligence or QA.

It is used only to:

- receive live microphone audio
- run the live room/session
- send spoken replies back

The real answer path remains:

```text
STT -> existing grounded QA pipeline -> TTS
```

## Repository Structure

```text
app/
  config/
    banks.yaml
    loader.py
    settings.py
  ingestion/
    discovery.py
    fetcher.py
    models.py
    normalizer.py
    parser.py
    pipeline.py
    storage.py
  chunking/
    cleaner.py
    chunker.py
  retrieval/
    indexer.py
    models.py
    retriever.py
    storage.py
    text.py
  guardrails/
    scope.py
  qa/
    evidence.py
    factory.py
    generator.py
    pipeline.py
    query_normalization.py
  prompts/
    grounded_qa.py
  voice/
    livekit_agent.py
    models.py
    service.py
    stt.py
    tts.py
  ui/
    gradio_app.py
scripts/
  scrape_all.py
  build_index.py
  test_retrieval.py
  test_qa.py
  test_voice_local.py
  test_gemini_tts.py
  run_voice_agent.py
  run_ui.py
  create_livekit_token.py
data/
  raw/
  processed/
  chunks/
  indexes/
tests/
```

## Setup Instructions

### 1. Create environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Then set at least:

```env
GEMINI_API_KEY=your_key_here
QA_GENERATOR=gemini
QA_GEMINI_MODEL=gemini-2.5-flash
```

The app reads `.env` automatically at runtime. `.env.example` is only a template.

### 3. Install LiveKit locally

```bash
brew install livekit
```

Then run local dev mode when needed:

```bash
livekit-server --dev
```

## How To Run The Main Pipeline

### Step 1. Scrape / ingest

Run all configured banks:

```bash
python scripts/scrape_all.py --run-id all_banks_strict_urls
```

Main output:

- `data/processed/all_banks_strict_urls/documents.jsonl`

### Step 2. Build chunks and retrieval index

```bash
python scripts/build_index.py \
  --run-id retrieval_best_clean \
  --documents data/processed/all_banks_strict_urls/documents.jsonl
```

Main output:

- `data/chunks/retrieval_best_clean/chunks.jsonl`
- `data/indexes/retrieval_best_clean/`

### Step 3. Test text QA

```bash
python scripts/test_qa.py \
  --index-dir data/indexes/retrieval_best_clean \
  --generator gemini \
  --query "Յունիբանկի սպառողական վարկի պայմանները ինչքա՞ն են"
```

### Step 4. Run the simple local interface

```bash
python scripts/run_ui.py --host 127.0.0.1 --port 8899
```

Open:

```text
http://127.0.0.1:8899
```

The UI supports:

- multiple chat turns without clearing history
- typed Armenian questions
- microphone/uploaded voice questions
- transcript display for the latest voice turn
- latest supporting sources
- latest spoken reply audio

### Step 5. Run the self-hosted LiveKit voice agent

In one terminal:

```bash
livekit-server --dev
```

In a second terminal:

```bash
python scripts/run_voice_agent.py dev
```

Create a local token:

```bash
python scripts/create_livekit_token.py --room bank-agent-room --identity voice-tester
```

Then join the local LiveKit room from a compatible client.

## Example Queries

### Text queries

```text
Յունիբանկի սպառողական վարկի պայմանները ինչքա՞ն են
ԱԵԲ բանկի ավանդի տոկոսադրույքը ինչքա՞ն է
ՎՏԲ-ի մասնաճյուղը որտեղ է Երևանում
```

### Clarification-style queries

```text
Վարկի պայմանները ինչքա՞ն են
Ավանդի տոկոսադրույքը ինչքա՞ն է
```

These may trigger a clarification request instead of a hard refusal if the bank is not specified.

### Refused queries

```text
Քարտի սպասարկման վճարը ինչքա՞ն է
Ո՞ր բանկի վարկն է ավելի լավ
```

## Environment Variables

Main runtime variables:

```env
GEMINI_API_KEY=
QA_GEMINI_MODEL=gemini-2.5-flash
QA_GENERATOR=gemini
QA_RETRIEVAL_TOP_K=6
QA_CONTEXT_CHUNKS=4
QA_MIN_TOP_SCORE=0.095
QA_MIN_TOTAL_SCORE=0.22

LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

VOICE_STT_PROVIDER=faster_whisper
VOICE_STT_MODEL=large-v3-turbo
VOICE_STT_LANGUAGE=hy
VOICE_STT_DEVICE=auto
VOICE_STT_COMPUTE_TYPE=int8
VOICE_STT_BEAM_SIZE=5

VOICE_TTS_PROVIDER=gemini_api_tts
VOICE_TTS_MODEL=gemini-2.5-flash-preview-tts
VOICE_TTS_LANGUAGE=hy-AM
VOICE_TTS_VOICE_NAME=
```

## Testing

Run tests:

```bash
pytest
```

Current test coverage includes:

- config loading
- parser/cleaner behavior
- chunking
- retrieval
- scope guard
- grounded QA pipeline
- voice service orchestration

## Known Limitations

- Gemini API TTS does not officially list `hy-AM` support right now, so Armenian speech is prompt-steered.
- Local STT quality depends on microphone/audio quality.
- Generic multi-bank questions may require clarification before the system answers.

## How To Push To GitHub

The folder is now initialized locally as a git repository, but there is still no GitHub remote configured.

If you want to publish it manually:

```bash
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Sources

- Gemini API speech generation: https://ai.google.dev/gemini-api/docs/speech-generation
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini API rate limits: https://ai.google.dev/gemini-api/docs/quota
- LiveKit local self-hosting: https://docs.livekit.io/home/self-hosting/local
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
