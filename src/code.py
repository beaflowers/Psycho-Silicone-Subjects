import time
import board
from digitalio import DigitalInOut, Direction, Pull

blue_btn = DigitalInOut(board.GP14)
blue_btn.direction = Direction.INPUT
blue_btn.pull = Pull.UP

yell_btn = DigitalInOut(board.GP15)
yell_btn.direction = Direction.INPUT
yell_btn.pull = Pull.UP

red_btn = DigitalInOut(board.GP16)
red_btn.direction = Direction.INPUT
red_btn.pull = Pull.UP

grn_btn = DigitalInOut(board.GP17)
grn_btn.direction = Direction.INPUT
grn_btn.pull = Pull.UP

while True:
    pressed_buttons = []

    if blue_btn.value:
        pressed_buttons.append("blue")
    if yell_btn.value:
        pressed_buttons.append("yellow")
    if red_btn.value:
        pressed_buttons.append("red")
    if grn_btn.value:
        pressed_buttons.append("green")

    if pressed_buttons:
        print(",".join(pressed_buttons))
    else:
        pass

    time.sleep(0.5)
