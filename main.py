import os
from signal import pause
import json
import time
import logging
import threading
import random
import colorsys
import functools
import serial
import gpiozero
import board, neopixel

from dotenv import load_dotenv

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))

load_dotenv()

class Rotary:
    def __init__(self, name, on_pressed, on_rotate, pin_a, pin_b, button_pin, direction=1):
        self.name = name
        self.button = gpiozero.Button(button_pin, pull_up=True)
        self.rot = gpiozero.RotaryEncoder(pin_a, pin_b, max_steps=0)
        self.direction = (direction == 1) and 1 or -1
        self.on_pressed = on_pressed and on_pressed or self._noop
        self.on_rotate = on_rotate and on_rotate or self._noop

        self.button.when_pressed = self._button_press
        self.rot.when_rotated = self._rotate

    def _noop(self):
        return

    def _rotate(self):
        self.on_rotate(self.name, self.rot.steps * self.direction)

    def _button_press(self):
        self.on_pressed(self.name)

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
                    self.update_func(json_data)
            except json.JSONDecodeError:
                logging.error("Received non-JSON data.")
            except Exception as e:
                logging.error(f"Error reading JSON: {e}")
            time.sleep(0.1)

    def send_json_message(self, data, refresh_state=False):
        try:
            if refresh_state:
                json_data['v'] = true
            json_data = json.dumps(data)
            self.ser.write((json_data).encode())
            logging.debug(f"Sent: {json_data}")
        except Exception as e:
            logging.debug(f"Error sending JSON: {e}")

    def read_json_message(self):
        try:
            if self.ser.in_waiting > 0:
                message = self.ser.readline().decode('utf-8').strip()
                json_data = json.loads(message)
                logging.debug(f"Received: {json_data}")
                return json_data
        except json.JSONDecodeError:
            logging.debug("Received non-JSON data.")
        except Exception as e:
            logging.debug(f"Error reading JSON: {e}")
        return None

class Main(object):
    def __init__(self):
        self.hue = 0
        self.hue_increment = float(os.getenv('HUE_INCREMENT', 0.05))

        # connect outputs
        self.led = neopixel.NeoPixel(getattr(board, os.getenv("LED_PIN")), int(os.getenv("LED_LENGTH")))
        self.wled = SerialInterface(os.getenv("SERIAL_PORT"), os.getenv("SERIAL_BAUD"), self.update_state)

        # ask for & wait for first state update
        self.got_state = threading.Event()
        self.state = {}
        logging.debug('requesting system state...')
        self.wled.send_json_message({"v": True})
        self.got_state.wait()

        # connect inputs
        self.encoder_a = Rotary('A', self.encoder_button_press, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_A")))
        self.encoder_a = Rotary('B', self.encoder_button_press, self.encoder_rotate, *self._scalarize(os.getenv("ENCODER_PINS_B")))

    def _scalarize(self, x):
        return [int(y.strip()) for y in x.split(',')]

    def encoder_button_press(self, encoder_name):
        logging.debug(f'button - {encoder_name}')

        if encoder_name == 'A':
            self.state['state']['on'] = not self.state['state']['on']
            self.wled.send_json_message({"on": self.state['state']['on']})

    def encoder_rotate(self, encoder_name, steps):
        logging.debug(f'rotate - {encoder_name} / {steps}')
        self.hue = (steps * self.hue_increment) % 1.0
        c = [int(x * 255) for x in colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)]
        self.wled.send_json_message({"tt": 0, "seg": [{"id": 0, "col": [c]}]})

    def update_state(self, data):
        self.state = data
        self.got_state.set()

    def run(self):
        pass

if __name__ == '__main__':
    m = Main()
    m.run()
