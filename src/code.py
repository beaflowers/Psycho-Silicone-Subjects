import time
import board
from digitalio import DigitalInOut, Direction, Pull

BUTTONS = {
    "blue": DigitalInOut(board.GP14),
    "yellow": DigitalInOut(board.GP15),
    "red": DigitalInOut(board.GP16),
    "green": DigitalInOut(board.GP17),
}

for button in BUTTONS.values():
    button.direction = Direction.INPUT
    button.pull = Pull.UP

last_pressed = ()

while True:
    pressed_buttons = tuple(
        name for name, button in BUTTONS.items() if button.value
    )

    if pressed_buttons and pressed_buttons != last_pressed:
        print(",".join(pressed_buttons))
        last_pressed = pressed_buttons
    elif not pressed_buttons:
        last_pressed = ()

    time.sleep(0.05)
