import os
import gc
import time
import logging
import threading
import atexit
import numpy as np
import pyaudio
import aubio

class NoteListener:
    def __init__(self, silence, callback):
        # initialise pyaudio
        self.start_audio()

        self.callback = callback

        # setup pitch
        tolerance = 0.8
        win_s = 4096  # fft size
        hop_s = self.buffer_size  # hop size

        self.note_o = aubio.notes("default", win_s, hop_s, self.samplerate)
        self.note_o.set_silence(silence)

        self.pitch_o = aubio.pitch("default", win_s, hop_s, self.samplerate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(tolerance)

        self.thread = threading.Thread(target=self.listen)
        self.thread.start()

    def start_audio(self):
        gc.collect()
        self.p = pyaudio.PyAudio()

        self.buffer_size = 512  # 1024
        self.pyaudio_format = pyaudio.paFloat32
        self.n_channels = 1
        self.samplerate = 44100
        logging.info('starting stream...')
        self.stream = self.p.open(format=self.pyaudio_format,
            channels=self.n_channels,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.buffer_size)
        while not self.stream.is_active():
            time.sleep(0.1)

    def stop_audio(self):
        logging.debug('stopping stream...')
        try:
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            while not self.stream.is_stopped():
                time.sleep(0.1)
            self.stream.close()
            self.p.terminate()
        except:
            self.stream = None
            self.p = None
        finally:
            gc.collect()

    def _midi_to_pitch(self, midi_number):
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = pitch_names[int(midi_number) % 12]
        octave = int((midi_number // 12) - 1)
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

            except Exception as e:
                logging.error(f"Error listening for notes: {e}")
                logging.debug('restarting audio...')
                self.stop_audio()
                self.start_audio()

    def cleanup(self):
        logging.debug('done recording, cleaning up...')
        self.stop_audio()
