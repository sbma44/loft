import gpiozero

class Rotary:
    def __init__(self, name, on_press, on_hold, on_rotate, pin_a, pin_b, button_pin, direction=1, hold_time=0.5):
        self.name = name
        self.button = gpiozero.Button(button_pin, hold_time=hold_time, pull_up=True)
        self.rot = gpiozero.RotaryEncoder(pin_a, pin_b, max_steps=0)
        self.direction = (direction == 1) and 1 or -1
        self.on_press = on_press and on_press or self._noop
        self.on_hold = on_hold and on_hold or self._noop
        self.on_rotate = on_rotate and on_rotate or self._noop

        self.button.when_pressed = self._button_press
        self.button.when_held = self._button_hold
        self.rot.when_rotated = self._rotate

    def _noop(self):
        return

    def _rotate(self):
        self.on_rotate(self.name, self.rot.steps * self.direction)

    def _button_press(self):
        self.on_press(self.name)

    def _button_hold(self):
        self.on_hold(self.name, self.button.held_time)