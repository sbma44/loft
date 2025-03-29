import os
import json
import time
import logging
import collections
import threading
import colorsys

import board
import neopixel

from dotenv import load_dotenv

from .io import NoteListener, Rotary, SerialInterface, ControlAnimator, WLEDRealtimeNoteEffect

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))

load_dotenv()

class Main:
    def __init__(self):
        self.last_action = None

        self.command_queue = collections.deque([], maxlen=50)
        self.mode = None
        self.segment_mode = False

        # set up notes, array to track state of each segment
        self.notes = os.getenv("NOTES", "").split(',')
        self.segment_colors = [-1] * len(self.notes)

        # pregenerate hues that note-playing will cycle through
        self.note_hues = []
        color_steps = int(os.getenv("NOTE_COLOR_STEPS", 7))
        hue_increment = 1.0 / color_steps
        current_hue = 0.0
        for i in range(len(self.notes)):
            self.note_hues.append(current_hue)
            current_hue += hue_increment

        # load enabled fx
        script_dir = os.path.dirname(os.path.abspath(__file__))
        selections_file = os.path.join(script_dir, 'effect_selections.json')
        self.enabled_fx = []
        with open(selections_file, 'r') as f:
            selected = json.load(f)
            for fx_i in selected:
                if selected[fx_i]:
                    self.enabled_fx.append(int(fx_i))

        # connect outputs
        self.led = neopixel.NeoPixel(getattr(board, os.getenv("LED_PIN")), int(os.getenv("LED_LENGTH")))
        self.wled = SerialInterface(os.getenv("SERIAL_PORT"), os.getenv("SERIAL_BAUD"), self.state_callback)
        self.control_animator = ControlAnimator(self.led)

        # reboot wled
        self.finished_reboot = threading.Event()
        self.wled.send_json_message({"rb": True})
        logging.info('waiting for WLED reboot...')
        self.finished_reboot.wait()

        # ask for & wait for first state update
        logging.debug('requesting system state...')
        self.got_state = threading.Event()
        self.state = {}
        self.power_on = False
        self.set_power_state(self.power_on)
        self.got_state.wait()

        self.hue = 0
        self.hue_increment = float(os.getenv('HUE_INCREMENT', 0.05))
        self.fx = self.state['state']['seg'][0]['fx']
        self.led_count = self.state['info']['leds']['count']
        self.leds_per_segment = int(self.led_count / len(self.notes))

        # realtime effect manager
        hues = list(range(0, 1.0, 1.0 / len(self.notes)))
        self.note_mode_output = WLEDRealtimeNoteEffect(os.getenv('WLED_HOSTNAME'), os.getenv('WLED_UDP_PORT'), hues=self.note_hues, leds_per_segment=self.leds_per_segment)
        self.note_mode_output.start()

        # connect inputs
        self.encoder_a = Rotary('A', self.encoder_button_press, self.encoder_button_hold, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_A")), float(os.getenv("ENCODER_BUTTON_HOLD_TIME", 0.5)))

        self.encoder_b = Rotary('B', self.encoder_button_press, self.encoder_button_hold, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_B")), float(os.getenv("ENCODER_BUTTON_HOLD_TIME", 0.5)))

        self.listener = NoteListener(float(os.getenv('SOUND_THRESHOLD_DB', -40)), self.note_callback)

        # send initial state to WLED
        self.set_mode('SOLID')
        self.send_color()

    def set_mode(self, mode):
        last_mode = self.mode
        if mode == last_mode:
            return

        logging.debug(f'setting mode to {mode}')
        self.mode = mode

        if self.mode == 'NOTE':
            self.note_mode_output.set_state('ACTIVE')
        else:
            if last_mode == 'NOTE':
                self.note_mode_output.set_state('SLEEP')

            if self.mode == 'SOLID':
                self.fx = 0
                self.send_fx()

    def log_command(self, command):
        if not self.power_on:
            return

        self.command_queue.appendleft(command)

        # only consider last 3 commands in the last 3 seconds
        lookback = 3
        time_threshold = time.time() - 3
        last_3s = [x for (i, x) in enumerate(self.command_queue) if i < lookback and x[1] > time_threshold]

        # if we don't have 3 commands in last 3s, ignore the following
        if len(last_3s) == lookback:
            # check for continuous events: note, solid, fx
            if all(item[0] == 'NOTE' for item in last_3s):
                self.set_mode('NOTE')
            # all rotation events?
            if all(item[0] == 'ROTATE' and item[2] == 'B' for item in last_3s):
                self.set_mode('FX')

    def _scalarize(self, x):
        return [int(y.strip()) for y in x.split(',')]

    def set_power_state(self, power_state):
        self.power_on = power_state
        animation = self.control_animator.get_solid_color_hsv([0, 0, 0])
        if not self.power_on:
            animation[2] = [5, 0, 0]
            animation[3] = [5, 0, 0]
        logging.debug(f'setting control color to {animation}')
        self.control_animator.set_animation([[animation, 0]], loop=False)
        self.wled.send_json_message({"on": self.power_on, "tt": 5, "v": True})

    def encoder_button_hold(self, encoder_name, held_time):
        logging.debug(f'hold - {encoder_name} for {held_time}s')

        if encoder_name == 'A':
            self.set_power_state(not self.power_on)

        self.log_command((f'HOLD', time.time(), encoder_name))

    def encoder_button_press(self, encoder_name):
        logging.debug(f'button - {encoder_name}')
        self.log_command((f'BUTTON', time.time(), encoder_name))
        if encoder_name == 'A':
            self.set_mode('SOLID')

    def encoder_rotate(self, encoder_name, steps, last_steps):
        logging.debug(f'rotate - {encoder_name} / {steps}')

        self.log_command((f'ROTATE', time.time(), encoder_name, steps))

        if encoder_name == 'A' and self.mode in ('SOLID', 'FX'):
            self.hue = (steps * self.hue_increment) % 1.0
            self.send_color()
        elif encoder_name == 'B' and self.mode == 'FX':
            fx_index = self.fx in self.enabled_fx and self.enabled_fx.index(self.fx) or 0
            self.fx = self.enabled_fx[(fx_index + (steps - last_steps)) % len(self.enabled_fx)]
            self.send_fx()

    def state_callback(self, state=False, non_state_message=False):
        if state:
            self.state = state
            self.got_state.set()
        elif non_state_message and non_state_message.strip() == 'Ada':
            self.finished_reboot.set()

    def note_callback(self, note, midi_note, pitch, confidence):
        logging.debug(f'note: {note}, midi: {midi_note}, pitch: {pitch}, confidence: {confidence}')

        self.log_command((f'NOTE', time.time(), note))

        if note in self.notes:
            note_index = len(self.notes) - (self.notes.index(note) + 1)
            self.note_mode_output.send_note(note_index % len(self.note_hues))

    def send_fx(self):
        self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "fx": self.fx}]})

    def send_color(self):
        c = [int(x * 255) for x in colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)]
        self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "start": 0, "stop": self.led_count - 1, "col": [c]}]})

if __name__ == '__main__':
    m = Main()