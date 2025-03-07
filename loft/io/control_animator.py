import time
import threading

class ControlAnimator:

    BLACK = [0, 0, 0]
    RED = [255, 0, 0]

    def __init__(self, leds):
        self.leds = leds
        self.pause = threading.Condition()

        self.blank()
        self.index = 0

        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def set_animation(self, animation, loop=True):
        self.index = 0
        self.animation = animation
        self.loop = loop

        # Acquire the condition before notifying
        with self.pause:
            self.pause.notify()

    def blank(self):
        accum = []
        for i in range(len(self.leds)):
            accum.append(self.BLACK)
        self.set_animation([[accum, 1.0]], loop=False)

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
                with self.pause:
                    self.pause.wait()
