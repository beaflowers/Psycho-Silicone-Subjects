"""Desktop entry point for reading Pico button data."""

from __future__ import annotations

from openai import APIError

## Importing from local modules
from .openai_client import create_client, get_model_name
from .pico_serial import open_pico_serial, read_line
from .prompting import build_prompt


def main() -> None:
    """Listen for Pico events, print button presses, and send them to OpenAI."""
    client = create_client()
    model = get_model_name()

    with open_pico_serial() as connection:
        print("Listening for Pico button data over USB serial...")

        while True:
            line = read_line(connection)
            print(f"Pressed: {line}")

            prompt = build_prompt(line)

            try:
                response = client.responses.create(model=model, input=prompt)
            except APIError as exc:
                print(f"OpenAI API error: {exc}")
                continue

            print("ChatGPT response:")
            print(response.output_text)
            print()


if __name__ == "__main__":
    main()
