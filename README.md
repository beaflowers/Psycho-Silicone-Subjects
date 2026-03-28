# AI Persona Simulation Studio
Interactive persona simulation with RAG memory, CLI tools, and a local website.
You can either chat with the persona or run a Milgram-style shock experiment with visible reasoning and session-level memory summaries.

## How it works now
- The app loads persona identity from `data/persona.json` (`age`, `job`, `social_hierarchy`, traits, rules).
- Memory data loads from `data/memories.jsonl`.
- Historical session summaries load from:
  - `data/chat_sessions.jsonl`
  - `data/shock_sessions.jsonl`
- Memories are embedded and indexed with FAISS at startup.
- Every turn retrieves prioritized memories (semantic relevance + recency + importance), with quotas across base/chat-session/shock-session memories.
- Chat mode returns:
  - in-character response
  - reasoning summary
  - memories used
- Experiment mode progresses authority commands by shock level and returns:
  - action (`obey` or `refuse`)
  - confidence
  - reasoning
  - memories used
- In web mode, `memories.jsonl` is treated as base persona memory.
- New web memories are saved only when you click:
  - `Finish Chat Session + Save Summary`
  - `Finish Shock Session + Save Summary`

## Project layout
```text
data/
  persona.json
  memories.jsonl
  chat_sessions.jsonl
  shock_sessions.jsonl
logs/
  runs.jsonl
  web_sessions/
src/ai_persona_sim/
  cli.py
  config.py
  models.py
  memory.py
  persona_engine.py
  decision_engine.py
  provider_openai.py
  logging_utils.py
  prompting.py
  web_app.py
  web/
    index.html
    app.js
    styles.css
pyproject.toml
.env.example
README.md
```

## Install
```bash
cd "/Users/uandha/Documents/New project/ai-persona-sim"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Configuration
Create env file:
```bash
cp .env.example .env
```

Set variables in `.env`:
```env
OPENAI_API_KEY=your_real_key_here
CHAT_MODEL=gpt-4.1
EMBED_MODEL=text-embedding-3-small
```

Optional model overrides:
```env
CHAT_MODEL=gpt-4.1-mini
EMBED_MODEL=text-embedding-3-large
```

## Run
CLI help:
```bash
persona-sim --help
```

Chat mode:
```bash
persona-sim chat --top-k 3
```

One-shot decision mode:
```bash
persona-sim decide "Press the button now" --shock-level 90
```

Simulation mode:
```bash
persona-sim simulate --steps 10
```

Website mode:
```bash
persona-sim web --host 127.0.0.1 --port 8080
```
Open:
```text
http://127.0.0.1:8080
```

## Website usage
- Start a session in the setup panel.
- Use `Chat` tab:
  - type user message
  - see response + reasoning + retrieved memories
  - click `Finish Chat Session + Save Summary` to save one summary record with visible rationale
- Use `Shock Experiment` tab:
  - set start/end/step
  - click `Start Experiment`
  - click `Run Next Step`
  - see authority command, level, action, confidence, reasoning, memory evidence
  - click `Finish Shock Session + Save Summary` to save one summary record with visible rationale
- In setup panel:
  - click `Refresh Summary History` to list saved summaries from `chat_sessions.jsonl` and `shock_sessions.jsonl`.
- Export traces as JSONL or CSV.

## Interactive controls
- In CLI `chat`, type `exit`, `quit`, or empty line to stop.
- In website experiment mode, the session ends on refusal or end level.

## Memory model
Memory schema in `data/memories.jsonl`:
- `id`
- `text`
- `valence` in `[-1, 1]`
- `intensity` in `[0, 1]`
- `relevance` in `[0, 1]` (optional, default `0.5`)
- `importance` in `[0, 1]` (optional)
- `created_at` ISO timestamp (optional)
- `source_type` (`base`, `chat_session`, `shock_session`, optional)
- `tags`

Priority retrieval score:
```text
0.55 * semantic_relevance + 0.25 * recency + 0.20 * importance
```

Persistence behavior:
- `chat`, `decide`, `simulate` persist memories by default.
- disable with `--no-persist-memories` in CLI.
- web mode does not append per-turn memories to `memories.jsonl`.
- web mode writes one summary per finished block to `chat_sessions.jsonl` / `shock_sessions.jsonl`.

## Logs
- CLI simulation trace: `logs/runs.jsonl`
- Website session traces: `logs/web_sessions/<session_id>.jsonl`

## Troubleshooting
`401 invalid_api_key`
- check `.env` has real `OPENAI_API_KEY`
- ensure you edited `.env` (not `.env.example`)

`429 tokens per min exceeded`
- lower `--top-k`
- start a fresh session
- reduce long memory history or disable persistence temporarily

`persona-sim: command not found`
```bash
source .venv/bin/activate
python -m pip install -e .
```

## Safety
This project is for simulation, learning, and research contexts only.
Do not use it to manipulate or harm real people.
