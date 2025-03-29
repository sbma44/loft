import json
import time
import logging
import threading
import serial

class SerialInterface:
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