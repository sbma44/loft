import time
import threading

class ControlAnimator:

    BLACK = [0, 0, 0]
    RED = [255, 0, 0]

    def __init__(self, leds):
        self.leds = leds

        self.blank()
        self.index = 0

        self.pause = threading.Condition()

        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def set_animation(self, animation, loop=True):
        self.index = 0
        self.animation = animation
        self.loop = loop
        self.pause.notify()

    def blank(self):
        self.set_animation([[[self.BLACK * len(self.leds)], 1.0]], loop=False)

    def _animate(self):
        while True:
            if self.index < len(self.animation):
                step = self.animation[self.index]
                led_states = step[0]
                delay = step[1]

                for i in range(len(led_states)):
                    self.leds[i] = led_states[i]

                self.index = self.index + 1
                if self.loop:
                    self.index = self.index % len(self.animation)

                time.sleep(delay)
            else:
                self.pause.wait()