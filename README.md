# Psycho-Silicone-Subjects

Final project for CART 498 Gen AI.

This repository contains multiple related apps built around persona-driven RAG systems (Jekyll/Hyde and Femwife), including web experiences and a Pico button-panel bridge.

## Active apps at a glance

- `milgram_experiment/`: shock-only web app (FastAPI + static web UI), default port `8010`
- `subject_chat/`: chat-only web app (FastAPI + static web UI), default port `8011`
- `jekyllandhyde/`: standalone Jekyll/Hyde RAG chat web app, default port `8000`
- `app/femandhousewife/`: terminal + Raspberry Pi Pico bridge for tone-shifted RAG

## Repo structure

- `milgram_experiment/` is the main shock-session experience
- `subject_chat/` is the main chat-session experience
- `jekyllandhyde/` is the standalone single-persona demo app
- `app/femandhousewife/` contains the Pico bridge and extra RAG experiments
- `local_backups/` stores archive snapshots
- `MilgramFullExperiment/` currently only contains diversion metadata (not an app runtime)

## Prerequisites

- Python 3
- `pip`
- OpenAI API key

## Quick start (web apps)

Run these from repo root.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r milgram_experiment/requirements.txt
pip install -r subject_chat/requirements.txt
pip install -r jekyllandhyde/requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r milgram_experiment/requirements.txt
pip install -r subject_chat/requirements.txt
pip install -r jekyllandhyde/requirements.txt
```

If PowerShell blocks activation, run this once per shell session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create env files (first time only).

macOS / Linux:

```bash
cp milgram_experiment/.env.example milgram_experiment/.env
cp subject_chat/.env.example subject_chat/.env
cp jekyllandhyde/.env.example jekyllandhyde/.env
```

Windows PowerShell:

```powershell
Copy-Item milgram_experiment/.env.example milgram_experiment/.env
Copy-Item subject_chat/.env.example subject_chat/.env
Copy-Item jekyllandhyde/.env.example jekyllandhyde/.env
```

Set `OPENAI_API_KEY` in each `.env`.

## Run each app

### 1) Shock experiment web app

macOS / Linux:

```bash
source .venv/bin/activate
uvicorn milgram_experiment.backend.server:app --reload --port 8010
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn milgram_experiment.backend.server:app --reload --port 8010
```

Open `http://127.0.0.1:8010`.

Main endpoints:
- `GET /api/health`
- `GET /api/personas`
- `POST /api/session/start`
- `POST /api/shock/pick-memory`
- `POST /api/shock/next`
- `POST /api/session/finish`
- `GET /api/session/{session_id}`

### 2) Subject chat web app

macOS / Linux:

```bash
source .venv/bin/activate
uvicorn subject_chat.backend.server:app --reload --port 8011
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn subject_chat.backend.server:app --reload --port 8011
```

Open `http://127.0.0.1:8011`.

Main endpoints:
- `GET /api/health`
- `GET /api/personas`
- `POST /api/session/start`
- `POST /api/chat/turn`
- `POST /api/session/finish`
- `GET /api/session/{session_id}`

### 3) Standalone Jekyll/Hyde web app

macOS / Linux:

```bash
source .venv/bin/activate
uvicorn jekyllandhyde.app.server:app --reload --port 8000
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn jekyllandhyde.app.server:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

If no PDFs are found automatically, set `RAG_PDF_PATHS` or `RAG_PDF_FOLDER` in `jekyllandhyde/.env`.

Main endpoints:
- `GET /api/health`
- `POST /api/chat`
- `POST /api/reset`

## Pico bridge (optional hardware mode)

Install dependencies and run:

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r app/femandhousewife/requirements.txt
cd app/femandhousewife
python -m src.pico_chatgpt_bridge.main
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r app/femandhousewife/requirements.txt
Set-Location app/femandhousewife
python -m src.pico_chatgpt_bridge.main
```

Useful env vars:
- `OPENAI_API_KEY` (required)
- `PICO_SERIAL_PORT` (optional, set if auto-detect fails)
- `RAG_PDF_FOLDER` or `RAG_PDF_PATHS` (optional PDF source controls)

## Data and outputs

Generated data is written in each app folder:

- `milgram_experiment/data/sessions/`
- `milgram_experiment/data/memories/`
- `milgram_experiment/data/logs/`
- `subject_chat/data/sessions/`
- `subject_chat/data/memories/`
- `subject_chat/data/logs/`

RAG caches are stored under `.rag_cache/` (repo root and/or app-level).

## Common issues

- Missing API key: ensure `OPENAI_API_KEY` is set in the correct app `.env`.
- Wrong port: each app has its own port (`8010`, `8011`, `8000`).
- Slow first response: first run builds/loads embedding cache.
