# Psycho-Silicone-Subjects

Interactive RAG bridge for a Raspberry Pi Pico button panel. The desktop app
now follows the notebook pattern more closely: you type a real question in the
terminal, the app retrieves relevant PDF chunks, and the Pico buttons act as
live tone modifiers instead of fixed prompt triggers.

## How it works now

1. Your Pico runs CircuitPython and reports the current pressed button state.
2. The desktop app listens for those button updates in the background.
3. You type a direct question in the terminal.
4. The last button you pressed is stored as the current tone mode, even after release.
5. The app retrieves relevant chunks from your PDF knowledge base.
6. OpenAI generates a grounded answer, optionally colored by the selected mode.

The RAG index is prepared at startup, so chunking, persona tagging, and
document embeddings happen before the first question instead of during the
first query.

Document chunks and embeddings are also cached locally in `.rag_cache/`, so the
PDF embedding cost is only paid again if the source PDFs or chunking settings
change.

## Shift-aware RAG

PDFs no longer need to live in one undifferentiated pool. Each chunk is tagged
with a `persona` during ingestion, then retrieval is blended between the two
archives using a continuous `shift` value:

- `shift = 0.0`: fully Angela Carter
- `shift = 1.0`: fully housewife
- values in between: mixed retrieval and a gradual voice transition

The most reliable setup is to group PDFs into persona-named subfolders:

```text
rag_docs/
  angela_carter/
    angela-carter-essays.pdf
  housewife/
    good-housewife-guide.pdf
```

Recursive PDF discovery is enabled, so those subfolders are indexed
automatically.

If you keep all PDFs in one folder, the app also tries to infer persona from
the filename. `angela`, `carter`, and `virago` map to `angela_carter`;
`housewife`, `homemaker`, `domestic`, and `housekeeping` map to `housewife`.

Conversation memory now persists while `shift` changes, so the model can keep
traces of the previous voice and drift gradually rather than resetting on each
persona change.

## Project layout

```text
src/code.py
src/RAG_femwife.ipynb
src/pico_chatgpt_bridge/main.py
src/pico_chatgpt_bridge/button_monitor.py
src/pico_chatgpt_bridge/rag_engine.py
src/pico_chatgpt_bridge/pico_serial.py
requirements.txt
src/pico_chatgpt_bridge_web/app.py
```

## Expected Pico serial messages

The Pico now sends button state changes like:

```text
blue
blue,red
none
```

`none` means all buttons are released. The desktop app now ignores `none` and
keeps the last non-empty button selection as the active mode.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Optionally set a serial port if auto-detection does not find your Pico:

```powershell
$env:PICO_SERIAL_PORT="COM5"
```

Point the RAG system at your PDF knowledge base. You can either use a folder:

```powershell
$env:RAG_PDF_FOLDER="C:\path\to\pdfs"
```

Or a list of explicit PDF paths separated by `;` on Windows:

```powershell
$env:RAG_PDF_PATHS="C:\docs\a.pdf;C:\docs\b.pdf"
```

Optional model overrides:

```powershell
$env:OPENAI_CHAT_MODEL="gpt-5-mini"
$env:OPENAI_EMBED_MODEL="text-embedding-3-large"
```

Optional retrieval drift tuning lives directly in
`src/pico_chatgpt_bridge/rag_engine.py`:

```python
SHIFT_AGGRESSION = 1.0
```

`1.0` keeps the content blend linear. Values above `1.0` make retrieval drift
more aggressively toward Angela or housewife as `shift` moves away from `0.5`.
Values below `1.0` make the transition gentler.

Optional cache controls:

```powershell
$env:RAG_DISABLE_CACHE="1"
```

Use that only when you want to force a rebuild of the local RAG cache.

You can also place the PDFs in `rag_docs/` at the project root and skip the
RAG path env vars.

## Run

```powershell
python -m src.pico_chatgpt_bridge.main
```

## Web prototype

A Flask-based mockumentary chat surface now lives at
`src/pico_chatgpt_bridge_web/`. It uses:

- OpenAI Conversations API for durable model-side thread state
- one local JSON chat-state file in `data/web_chat_state.json`
- the existing RAG and shift/memory prompt helpers from the desktop app

Run it from the repo root with:

```powershell
python -m src.pico_chatgpt_bridge_web.app
```

Then open `http://127.0.0.1:5000/` in your browser.

## Interactive commands

- Type a question and press Enter to query the RAG system.
- Type `/reset` to clear conversation memory.
- Type `/shift` or `/shift current` to show the current blend.
- Type `/shift 0` for Angela Carter retrieval and voice.
- Type `/shift 1` for housewife retrieval and voice.
- Type `/shift 0.35` for a gradual blend.
- Type `/shift angela` or `/shift housewife` as shortcuts.
- Press Enter on an empty line, or type `quit` / `exit`, to close the app.

## Button tones

- `blue`: melancholy and introspective
- `yellow`: cheerful and optimistic
- `red`: angry and aggressive
- `green`: calm and peaceful

Pressing a button once latches that mode until another button is pressed.

If no Pico is connected, the app still runs in interactive RAG mode without
button tone control.
