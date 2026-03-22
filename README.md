# Psycho-Silicone-Subjects

Starter structure for reading button data from a Raspberry Pi Pico over USB
serial using Python on your computer, then sending that button value to the
OpenAI API.

## Project layout

```text
src/pico_chatgpt_bridge/main.py
src/pico_chatgpt_bridge/pico_serial.py
requirements.txt
```

## How the data flows

1. Your Pico runs CircuitPython and reads the four buttons.
2. The Pico sends a text line over USB serial.
3. The Python app on your computer reads that line.
4. The app prints the button name or names that were sent.
5. The app builds a prompt from that button value and sends it to OpenAI.

## Expected serial message

The desktop app expects each line from the Pico to look like this:

```text
blue
```

If more than one button is pressed, the Pico can send a comma-separated line:

```text
blue,red
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Optionally set a serial port if auto-detection does not find your Pico:

```powershell
$env:PICO_SERIAL_PORT="COM5"
```

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

You can also use a local `.env` file in the project root instead of setting
PowerShell variables every time. This is the recommended option. Copy
`.env.example` to `.env` and add your real key:

```env
OPENAI_API_KEY=your_api_key_here
PICO_SERIAL_PORT=COM8
```

The app will load `.env` automatically at startup.

## Model Choice

The model is chosen directly in the code so it stays visible. You can change it
in [openai_client.py](/c:/Users/b/Documents/Concordia/CART498-GenAI/Final%20Project/Repo/Psycho-Silicone-Subjects/src/pico_chatgpt_bridge/openai_client.py#L11).

## Run

```powershell
python -m src.pico_chatgpt_bridge.main
```

When the Pico sends button data, the host app will print the button name or
names it receives, then print the model response.
