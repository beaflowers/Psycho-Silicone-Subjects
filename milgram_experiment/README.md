# Experiment Session (Shock Web App)

This app is the shock-only web application split from the Milgram workspace.

It uses the two existing RAG personalities:
- `jekyllhyde` from `jekyllandhyde/app/rag_engine.py`
- `femwife` from `app/femandhousewife/src/pico_chatgpt_bridge/rag_engine.py`

Chat-only functionality now lives in `subject_chat`.

## Features
- Shock experiment flow with admin and receiver personas
- Per-level authority command handling (`/api/shock/next`)
- Start/finish session controls
- Per-session transcript JSON
- Per-session per-persona memory JSON
- End-of-session reflection memories for each participant

## Run
From repo root (`/Users/uandha/Milgram_Silicon_Subjects`):

```bash
cd milgram_experiment
cp .env.example .env
# put your real OPENAI_API_KEY in .env

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ..
uvicorn milgram_experiment.backend.server:app --reload --port 8010
```

Then open: `http://127.0.0.1:8010`

## Data files
- Session files: `milgram_experiment/data/sessions/<session_id>.json`
- Memory files: `milgram_experiment/data/memories/<session_id>_<persona>.json`

## API overview
- `GET /api/health`
- `POST /api/session/start`
- `POST /api/shock/next`
- `POST /api/session/finish`
- `GET /api/session/{session_id}`
