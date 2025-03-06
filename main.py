import os
from signal import pause
import json
import time
import logging
import threading
import atexit
import random
import colorsys
import functools
import serial
import gpiozero
import board, neopixel
import pyaudio
import numpy as np
import aubio

from dotenv import load_dotenv

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))

load_dotenv()

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

class SerialInterface(object):
    def __init__(self, port, baud, state_update_func=None):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.update_func = state_update_func
        if callable(self.update_func):
            logging.debug('starting serial watcher')
            self.watch_thread = threading.Thread(target=self._watch_for_state_updates)
            self.watch_thread.start()

    def _watch_for_state_updates(self):
        while True:
            try:
                # Read data from the ESP32
                if self.ser.in_waiting > 0:
                    message = self.ser.readline().decode('utf-8').strip()
                    # Parse the message as JSON
                    json_data = json.loads(message)
                    logging.debug(f"Received: {json_data}")
                    self.update_func(state=json_data)
            except json.JSONDecodeError:
                logging.error(f"Received non-JSON data: {message}")
                self.update_func(non_state_message=message)
            except Exception as e:
                logging.error(f"Error reading JSON: {e}")
            time.sleep(0.1)

    def send_json_message(self, data, refresh_state=False):
        try:
            if refresh_state:
                data['v'] = True
            json_data = json.dumps(data) + '\n'
            logging.debug(f"Sending: {json_data.strip()}")
            self.ser.write((json_data).encode())
        except Exception as e:
            logging.debug(f"Error sending JSON: {e}")

class NoteListener(object):
    def __init__(self, silence, callback):
        # initialise pyaudio
        self.p = pyaudio.PyAudio()
        self.callback = callback

        # open stream
        self.buffer_size = 1024
        pyaudio_format = pyaudio.paFloat32
        n_channels = 1
        samplerate = 44100
        self.stream = self.p.open(format=pyaudio_format,
                        channels=n_channels,
                        rate=samplerate,
                        input=True,
                        frames_per_buffer=self.buffer_size)

        # setup pitch
        tolerance = 0.8
        win_s = 4096 # fft size
        hop_s = self.buffer_size # hop size

        self.note_o = aubio.notes("default", win_s, hop_s, samplerate)
        self.note_o.set_silence(silence)

        self.pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(tolerance)

        self.thread = threading.Thread(target=self.listen)
        self.thread.start()

    def _midi_to_pitch(self, midi_number):
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = pitch_names[int(midi_number) % 12]
        octave = (midi_number // 12) - 1
        return f"{note}{octave}"

    def listen(self):
        atexit.register(self.cleanup)

        while True:
            try:
                audiobuffer = self.stream.read(self.buffer_size)
                signal = np.fromstring(audiobuffer, dtype=np.float32)

                note = self.note_o(signal)
                pitch = self.pitch_o(signal)[0]
                confidence = self.pitch_o.get_confidence()

                if note[0] > 0:
                    self.callback(self._midi_to_pitch(note[0]), note, pitch, confidence)
            except:
                pass

    def cleanup(self):
        logging.debug('done recording, cleaning up...')
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class Main(object):
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
        self.encoder_a = Rotary('B', self.encoder_button_press, self.encoder_button_hold, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_B")), float(os.getenv("ENCODER_BUTTON_HOLD_TIME", 0.5)))
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
