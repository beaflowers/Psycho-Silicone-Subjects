# Interview Room (Pre + Post Experiment)

This app runs both pre-experiment and post-experiment interview workflows.

- Interviewed persona: `jekyllhyde` or `femwife` (RAG-backed)
- Interviewer: dedicated interviewer model (non-RAG), one question at a time

## Features

- Start a pre-experiment session
- Start a post-experiment session (auto-loads one source shock experience per interview, rotating by persona)
- Interview loop:
  - interviewed answers current question with reasoning
  - interviewer reflects and generates the next non-repeated question
- Prompt debug panels for both sides
- Per-session transcript JSON
- Role-separated memory files (`interviewer` / `interviewed`)
- Final reflections for both roles

## Run

From repo root (`/Users/uandha/Milgram_Silicon_Subjects`):

```bash
cd interview_room
cp .env.example .env
# set OPENAI_API_KEY in .env
# optional: set INTERVIEWER_MODEL in .env

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ..
uvicorn interview_room.backend.server:app --reload --port 8012
```

Open: `http://127.0.0.1:8012`

## Environment Variables

- `OPENAI_API_KEY` (required)
- `INTERVIEWER_MODEL` (optional, interviewer-only model)
- `MILGRAM_JEKYLL_MODEL` (optional, interviewed J/H RAG model)
- `MILGRAM_FEMWIFE_MODEL` (optional, interviewed F/W RAG model)

## Data Files

- Canonical session JSON:
  - `interview_room/data/sessions/<session_id>.json`
- Pre-experiment mirrors:
  - `interview_room/data/pre_experiment/sessions/<session_id>.json`
  - `interview_room/data/pre_experiment/memories/<session_id>_interviewer.json`
  - `interview_room/data/pre_experiment/memories/<session_id>_interviewed.json`
- Post-experiment mirrors:
  - `interview_room/data/post_experiment/sessions/<session_id>.json`
  - `interview_room/data/post_experiment/memories/<session_id>_interviewer.json`
  - `interview_room/data/post_experiment/memories/<session_id>_interviewed.json`
- Post source inputs:
  - `interview_room/data/sessions_post/*.json`
  - `interview_room/data/memories_post/shock_*_<persona>.json`

## API Overview

- `GET /api/health`
- `GET /api/personas`
- `POST /api/pre-experiment/start`
- `POST /api/pre-experiment/next`
- `POST /api/post-experiment/start`
- `POST /api/post-experiment/next`
- `POST /api/session/finish`
- `GET /api/session/{session_id}`
