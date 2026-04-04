# Subject Chat (Web)

This app is the chat-only split from the Milgram workspace.

## Features
- One selected persona per session
- Multi-turn chat with memory continuity
- Session transcript JSON
- Per-session memory JSON
- End-of-session reflection

## Run
From repo root (`/Users/uandha/Milgram_Silicon_Subjects`):

```bash
cd subject_chat
cp .env.example .env
# put your real OPENAI_API_KEY in .env

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ..
uvicorn subject_chat.backend.server:app --reload --port 8011
```

Then open: `http://127.0.0.1:8011`

## Data files
- Session files: `subject_chat/data/sessions/<session_id>.json`
- Memory files: `subject_chat/data/memories/<session_id>_<persona>.json`

## API overview
- `GET /api/health`
- `POST /api/session/start`
- `POST /api/chat/turn`
- `POST /api/session/finish`
- `GET /api/session/{session_id}`
