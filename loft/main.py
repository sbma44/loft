import os
import json
import time
import logging
import threading
import colorsys
import board
import neopixel
from dotenv import load_dotenv

from .io import NoteListener, Rotary, SerialInterface

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))

load_dotenv()

class Main:
    def __init__(self):
        self.last_action = None

        self.notes = os.getenv("NOTES", "").split(',')
        self.segment_colors = [0] * len(self.notes)
        self.colors = []
        color_steps = int(os.getenv("NOTE_COLOR_STEPS", 7))

        for i in range(color_steps):
            self.colors.append([int(x * 255) for x in colorsys.hsv_to_rgb(i / color_steps, 1.0, 1.0)])

        # connect outputs
        self.led = neopixel.NeoPixel(getattr(board, os.getenv("LED_PIN")), int(os.getenv("LED_LENGTH")))
        self.wled = SerialInterface(os.getenv("SERIAL_PORT"), os.getenv("SERIAL_BAUD"), self.update_state)

        # reboot wled
        self.finished_reboot = threading.Event()
        self.wled.send_json_message({"rb": True})
        logging.info('waiting for WLED reboot...')
        self.finished_reboot.wait()

        # ask for & wait for first state update
        self.got_state = threading.Event()
        self.state = {}
        logging.debug('requesting system state...')
        self.wled.send_json_message({"on": False, "v": True})
        self.got_state.wait()

        self.hue = 0
        self.hue_increment = float(os.getenv('HUE_INCREMENT', 0.05))
        self.fx = self.state['state']['seg'][0]['fx']

        self.led_count = self.state['info']['leds']['count']
        self.leds_per_segment = int(self.led_count / len(self.notes))

        # connect inputs
        self.encoder_a = Rotary('A', self.encoder_button_press, self.encoder_button_hold, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_A")), float(os.getenv("ENCODER_BUTTON_HOLD_TIME", 0.5)))
        self.encoder_b = Rotary('B', self.encoder_button_press, self.encoder_button_hold, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_B")), float(os.getenv("ENCODER_BUTTON_HOLD_TIME", 0.5)))
        self.listener = NoteListener(float(os.getenv('SOUND_THRESHOLD_DB', -40)), self.note_callback)

    def _scalarize(self, x):
        return [int(y.strip()) for y in x.split(',')]

    # erase all segment data so we can manipulate the whole string. it will be recreated when notes arrive.
    def _clear_segments(self):
        if self.last_action == 'NOTE':
            seg = []
            for i in range(1, len(self.notes) + 1):
                seg.append({"id": i, "start": 1, "stop": 0 })
            self.wled.send_json_message({"seg": seg})

    def encoder_button_hold(self, encoder_name, held_time):
        logging.debug(f'hold - {encoder_name} for {held_time}s')

        if encoder_name == 'A':
            self.state['state']['on'] = not self.state['state']['on']
            self.wled.send_json_message({"on": self.state['state']['on'], "tt": 5})
            self.last_action = 'POWER_TOGGLE'

    def encoder_button_press(self, encoder_name):
        logging.debug(f'button - {encoder_name}')
        self._clear_segments()
        if encoder_name == 'A':
            self.fx = 0
            self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "fx": self.fx}]})
            self.last_action = 'SET_SOLID_COLOR'

    def encoder_rotate(self, encoder_name, steps):
        logging.debug(f'rotate - {encoder_name} / {steps}')
        self._clear_segments()
        if encoder_name == 'A':
            self.hue = (steps * self.hue_increment) % 1.0
            c = [int(x * 255) for x in colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)]
            self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "start": 0, "stop": self.led_count - 1, "col": [c]}]})
            self.last_action = 'INCREMENT_COLOR'
        elif encoder_name == 'B':
            self.fx = (self.fx + 1) % self.state['info']['fxcount']
            self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "fx": self.fx}]})
            self.last_action = 'INCREMENT_FX'

    def update_state(self, state=False, non_state_message=False):
        if state:
            self.state = state
            self.got_state.set()
        elif non_state_message and non_state_message.strip() == 'Ada':
            self.finished_reboot.set()

    def note_callback(self, note, midi_note, pitch, confidence):
        logging.debug(f'note: {note}, midi: {midi_note}, pitch: {pitch}, confidence: {confidence}')
        if note in self.notes:
            # check if we've been hearing notes; if not, blank out the strip
            if self.last_action != 'NOTE':
                self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "fx": 0, "col": [[0, 0, 0]]}]})

            note_index = len(self.notes) - (self.notes.index(note) + 1)
            self.segment_colors[note_index] = (self.segment_colors[note_index] + 1) % len(self.colors)

            segment_start = self.leds_per_segment * note_index
            segment_end = segment_start + self.leds_per_segment
            self.wled.send_json_message({"tt": 0, "seg": [{"id": note_index + 1, "start": segment_start, "stop": segment_end, "col": [self.colors[self.segment_colors[note_index]]]}]})
            self.last_action = 'NOTE'

if __name__ == '__main__':
    m = Main() 