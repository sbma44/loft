import os, os.path
import sys
import gc
import time
import logging
import threading
import atexit
import numpy as np
import pyaudio
import aubio
import joblib

from dotenv import load_dotenv
load_dotenv()

class NoteListener:
    def __init__(self, min_volume, callback):
        self.min_volume = min_volume
        self.callback = callback

        self.buffer_size = int(os.getenv('BUFFER_SIZE', 512))  # or 1024?
        self.pyaudio_format = pyaudio.paFloat32
        self.n_channels = 1
        self.samplerate = int(os.getenv('SAMPLE_RATE', 44100))
        self.instrument_listen_threshold = float(os.getenv('INSTRUMENT_LISTEN_THRESHOLD', 3.0))

        # initialise pyaudio
        self.start_audio()

        # setup pitch
        tolerance = 0.8
        win_s = 4096  # fft size
        hop_s = self.buffer_size  # hop size

        self.instrument_classifier = joblib.load(os.getenv('MODEL_PATH'))

        self.note_o = aubio.notes("default", win_s, hop_s, self.samplerate)
        self.pitch_o = aubio.pitch("default", win_s, hop_s, self.samplerate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(tolerance)

        self.thread = threading.Thread(target=self.listen)
        self.thread.start()

    @staticmethod
    def extract_features(buffer, samplerate):
        windowed = buffer * np.hamming(len(buffer))
        spectrum = np.abs(np.fft.rfft(windowed))

        # Avoid log of zero
        spectrum = np.where(spectrum == 0, 1e-10, spectrum).reshape(-1, 1)

        n_fft = len(buffer)
        centroid = librosa.feature.spectral_centroid(S=spectrum, sr=samplerate, n_fft=n_fft)[0, 0]
        bandwidth = librosa.feature.spectral_bandwidth(S=spectrum, sr=samplerate, n_fft=n_fft)[0, 0]
        flatness = librosa.feature.spectral_flatness(S=spectrum, n_fft=n_fft)[0, 0]
        rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=samplerate, n_fft=n_fft)[0, 0]
        rms = np.sqrt(np.mean(buffer**2))
        zcr = np.mean(librosa.feature.zero_crossing_rate(buffer)[0])

        return [centroid, bandwidth, flatness, rolloff, rms, zcr]

    @staticmethod
    def get_volume_db(buffer):
        volume = np.sqrt(np.mean(buffer ** 2))
        if volume <= 0:
            return -100.0  # Return a very low dB value for silence if zero
        # Calculate dB relative to full scale (maximum possible amplitude)
        db = 20 * np.log10(volume / 1.0)
        # Clip to reasonable range (-100 dB to 0 dB)
        return max(-100, min(0, db))

    def start_audio(self):
        gc.collect()
        self.p = pyaudio.PyAudio()

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

        valid_instruments = ('glockenspiel',)
        last_instrument_detection = 0

        while True:
            try:
                audiobuffer = self.stream.read(self.buffer_size)
                signal = np.fromstring(audiobuffer, dtype=np.float32)

                volume = self.get_volume_db(signal)
                if volume > self.min_volume:

                    # test for glockenspiel-y-ness and activate a 3s wake period
                    features = self.extract_features(signal, self.samplerate)
                    prediction = self.instrument_classifier.predict([features])[0]
                    if prediction == 'glockenspiel':
                        last_instrument_detection = time.time()

                    # are we in the 3s wake period? if so, do note detection
                    if time.time() - last_instrument_detection < self.instrument_listen_threshold:
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
