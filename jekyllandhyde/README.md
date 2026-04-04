# Dr Jekyll & Mr Hyde RAG Web Chat (VS Code)

This project converts your Colab RAG flow into a local web application.

## What it includes

- `app/rag_engine.py`: PDF load, chunk, embed, retrieve, answer
- `app/server.py`: FastAPI backend (`/api/chat`, `/api/reset`, `/api/health`)
- `web/index.html`: browser chat interface

The notebook `DrJekyll_and_Mr_Hyde_RAG.ipynb` is left unchanged.

## 1) Setup in VS Code terminal

```bash
cd /Users/uandha/Milgram_Silicon_Subjects/jekyllandhyde
source /Users/uandha/Milgram_Silicon_Subjects/.venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure environment

`.env` is already created in this folder.

Edit `.env` and set:

- `OPENAI_API_KEY`
- optionally adjust `RAG_PDF_PATHS` if you want a different PDF path

By default, the app auto-tries:

1. Any `*.pdf` in this project folder (uses the first match)
2. Jekyll/Hyde-like filenames in `~/Downloads` (uses the first match)

For full control, set `RAG_PDF_PATHS` explicitly.

## 3) Run server

```bash
uvicorn app.server:app --reload
```

Open: `http://127.0.0.1:8000`

## Notes

- First question may take longer because embeddings are built.
- Cache files are saved in `.rag_cache/`.
- Use **Reset** in the UI to clear conversation memory.
