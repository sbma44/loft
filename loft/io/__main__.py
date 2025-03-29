#!/usr/bin/env python3
"""
Command-line interface for loft.io operations.

This module provides commands for data collection, calibration, and training
of the Loft audio-reactive LED controller.

Usage:
    python -m loft.io COMMAND [OPTIONS]

Commands:
    train           - Train a model using collected audio samples
    collect         - Record audio samples for training the model
    calibrate       - Calibrate the audio input levels
    test-realtime   - Test the realtime note effect
    help            - Show this help message
"""

import os
import sys
import time
import glob
import logging
import argparse
from collections import defaultdict
from importlib import import_module
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('loft.io')

# Load samples and labels from WAV files (assumes a dict: {label: [wav_paths]})
def load_samples_from_wavs(labelled_files):
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 44100))
    BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', 512))
    VOLUME_THRESHOLD = float(os.getenv('SOUND_THRESHOLD_DB', -50))
    X, y = [], []
    for label, paths in labelled_files.items():
        for path in paths:
            samples, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            for i in range(0, len(samples) - BUFFER_SIZE, BUFFER_SIZE):
                buffer = samples[i:i + BUFFER_SIZE]
                if len(buffer) < BUFFER_SIZE:
                    continue
                # ignore silence
                if NoteListener.get_volume_db(buffer) < VOLUME_THRESHOLD:
                    continue
                features = NoteListener.extract_features(buffer, SAMPLE_RATE)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Train and evaluate the model
def train_model(labelled_files):
    MODEL_PATH = os.getenv('MODEL_PATH')
    X, y = load_samples_from_wavs(labelled_files)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
def train_command(args):
    """Train a model using collected audio samples."""
    try:
        # Dynamically import the necessary modules to avoid import errors
        # if non-required dependencies are missing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import joblib

        # Import local functions
        from .note_listener import NoteListener
    except ImportError as e:
        logger.error(f"Missing dependencies for training: {e}")
        logger.error("Install required packages with: pip install scikit-learn joblib")
        return 1

    logger.info("Starting model training...")

    training_data_directory = os.getenv('TRAINING_DATA_DIRECTORY')
    if not training_data_directory:
        logger.error("TRAINING_DATA_DIRECTORY environment variable not set")
        return 1

    if not os.path.exists(training_data_directory):
        logger.error(f"Training data directory does not exist: {training_data_directory}")
        return 1

    # Find all training files
    sample_files = defaultdict(list)
    for filename in glob.glob(os.path.join(training_data_directory, '*.wav')):
        try:
            label, remainder = os.path.basename(filename).split('_', 1)
            sample_files[label].append(filename)
        except ValueError:
            logger.warning(f"Skipping file with invalid naming format: {filename}")

    if not sample_files:
        logger.error(f"No training files found in {training_data_directory}")
        return 1

    logger.info(f"Found {sum(len(files) for files in sample_files.values())} files across {len(sample_files)} classes")

    # Train model
    model_path = os.getenv('MODEL_PATH')
    if not model_path:
        logger.error("MODEL_PATH environment variable not set")
        return 1

    try:
        X, y = load_samples_from_wavs(sample_files)
        if len(X) == 0:
            logger.error("No valid samples extracted from audio files")
            return 1

        logger.info(f"Extracted {len(X)} samples for training")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("\nClassification report:")
        logger.info(classification_report(y_test, preds))

        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)
        logger.info(f"Model saved to {model_path}")

        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

def collect_command(args):
    """Record audio samples for training the model."""
    try:
        import pyaudio
        import wave
        import keyboard
    except ImportError as e:
        logger.error(f"Missing dependencies for data collection: {e}")
        logger.error("Install required packages with: pip install pyaudio wave keyboard")
        return 1

    logger.info("Starting audio collection...")

    training_data_directory = os.getenv('TRAINING_DATA_DIRECTORY')
    if not training_data_directory:
        logger.error("TRAINING_DATA_DIRECTORY environment variable not set")
        return 1

    # Create directory if it doesn't exist
    os.makedirs(training_data_directory, exist_ok=True)

    # Audio recording settings
    sample_rate = int(os.getenv('SAMPLE_RATE', '44100'))
    buffer_size = int(os.getenv('BUFFER_SIZE', '512'))
    record_seconds = args.duration if hasattr(args, 'duration') else 5

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=buffer_size
    )

    logger.info(f"Recording {record_seconds} seconds of audio for label '{args.label}'")
    logger.info("Press 'ESC' to stop recording early")

    # Record data
    frames = []
    start_time = time.time()

    try:
        while time.time() - start_time < record_seconds:
            if keyboard.is_pressed('esc'):
                logger.info("Recording stopped early")
                break

            data = stream.read(buffer_size)
            frames.append(data)

            # Show progress
            progress = (time.time() - start_time) / record_seconds * 100
            sys.stdout.write(f"\rRecording: {progress:.1f}%")
            sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        return 1
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Generate filename with timestamp
    timestamp = int(time.time())
    filename = os.path.join(training_data_directory, f"{args.label}_{timestamp}.wav")

    # Save the recorded data as a WAV file
    try:
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        logger.info(f"\nAudio saved to {filename}")
        return 0
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        return 1

def calibrate_command(args):
    """Calibrate the audio input levels."""
    try:
        import pyaudio
        import numpy as np
        import keyboard
    except ImportError as e:
        logger.error(f"Missing dependencies for calibration: {e}")
        logger.error("Install required packages with: pip install pyaudio numpy keyboard")
        return 1

    logger.info("Starting audio calibration...")
    logger.info("Make noise at the levels you want to detect")
    logger.info("Press 'ESC' to stop calibration")

    # Audio settings
    sample_rate = int(os.getenv('SAMPLE_RATE', '44100'))
    buffer_size = int(os.getenv('BUFFER_SIZE', '512'))

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=buffer_size
    )

    # Function to calculate dB from audio buffer
    def get_volume_db(buffer):
        buffer_array = np.frombuffer(buffer, dtype=np.float32)
        volume = np.sqrt(np.mean(buffer_array ** 2))
        if volume <= 0:
            return -100.0
        db = 20 * np.log10(volume / 1.0)
        return max(-100, min(0, db))

    # Collect volume data
    volume_data = []

    try:
        while True:
            if keyboard.is_pressed('esc'):
                break

            data = stream.read(buffer_size)
            db_level = get_volume_db(data)
            volume_data.append(db_level)

            # Show current level
            bar_length = 50
            bar = '=' * int((db_level + 100) * bar_length / 100)
            sys.stdout.write(f"\rLevel: {db_level:.1f} dB [{bar.ljust(bar_length)}]")
            sys.stdout.flush()

            time.sleep(0.05)
    except Exception as e:
        logger.error(f"Error during calibration: {e}")
        return 1
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    if volume_data:
        min_db = min(volume_data)
        max_db = max(volume_data)
        avg_db = sum(volume_data) / len(volume_data)

        logger.info("\n\nCalibration Results:")
        logger.info(f"Minimum level: {min_db:.1f} dB")
        logger.info(f"Maximum level: {max_db:.1f} dB")
        logger.info(f"Average level: {avg_db:.1f} dB")
        logger.info("\nRecommended threshold settings:")

        # Recommend a value between min and max, but closer to min
        recommended = min_db + (max_db - min_db) * 0.25
        logger.info(f"SOUND_THRESHOLD_DB={recommended:.1f}  # Add this to your .env file")

        return 0
    else:
        logger.error("No data collected during calibration")
        return 1

def test_realtime_command(args):
    """Test the realtime note effect."""
    try:
        # Try to import from wled_realtime_note_effect, but fall back to realtime_note_effect if needed
        try:
            from .wled_realtime_note_effect import WLEDRealtimeNoteEffect as RealtimeNoteEffect
        except ImportError:
            try:
                from .realtime_note_effect import RealtimeNoteEffect
            except ImportError:
                logger.error("Could not import RealtimeNoteEffect class")
                return 1

        # Get hostname from args or environment
        hostname = args.hostname if hasattr(args, 'hostname') else os.getenv('WLED_HOSTNAME', 'wled-lamp.local')
        udp_port = int(os.getenv('WLED_UDP_PORT', '21324'))

        # Create some evenly spaced hues based on the number of segments
        num_segments = args.segments if hasattr(args, 'segments') else 5
        hues = [i/num_segments for i in range(num_segments)]
        leds_per_segment = args.leds_per_segment if hasattr(args, 'leds_per_segment') else 16

        logger.info(f"Testing realtime note effect with {num_segments} segments")
        logger.info(f"Hostname: {hostname}, UDP Port: {udp_port}")

        # Create and start the effect
        effect = RealtimeNoteEffect(hostname, udp_port, hues, leds_per_segment)
        logger.info(f"UDP Port: {effect.udp_port}")
        logger.info(f"Note Decay: {effect.note_decay_s}s")
        logger.info(f"Max FPS: {effect.max_fps}")

        effect.start()
        effect.set_state('ACTIVE')

        print("\nInteractive Note Tester")
        print("======================")
        print("Press 1-{} to trigger notes".format(num_segments))
        print("Press 'q' to exit")

        # Function to get a key press without requiring Enter
        def get_key():
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        try:
            while True:
                key = get_key()

                # Quit on 'q'
                if key.lower() == 'q':
                    print("\nExiting...")
                    break

                # Try to convert key to a segment index (1-based input, 0-based index)
                try:
                    note_idx = int(key) - 1
                    if 0 <= note_idx < num_segments:
                        print(f"Triggering note {note_idx} (hue: {hues[note_idx]:.2f})")
                        effect.send_note(note_idx)
                except ValueError:
                    # Not a number, ignore
                    pass
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            effect.stop()

        return 0
    except Exception as e:
        logger.error(f"Error in test-realtime command: {e}", exc_info=True)
        return 1

def show_help():
    """Show the help message."""
    print(__doc__)
    return 0

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='Loft IO command-line interface',
        add_help=False
    )

    # Add main command argument
    parser.add_argument('command', nargs='?', default='help',
                      help='Command to execute')

    # First parse just the command
    args, remaining_args = parser.parse_known_args()

    # Map commands to functions
    commands = {
        'train': train_command,
        'collect': collect_command,
        'calibrate': calibrate_command,
        'test-realtime': test_realtime_command,
        'help': lambda _: show_help(),
    }

    # If the command exists, add its specific arguments
    if args.command in commands:
        if args.command == 'collect':
            parser = argparse.ArgumentParser(description='Collect audio samples')
            parser.add_argument('command', help='Command name')
            parser.add_argument('label', help='Label for the audio sample')
            parser.add_argument('--duration', type=int, default=5,
                              help='Recording duration in seconds (default: 5)')
        elif args.command == 'test-realtime':
            parser = argparse.ArgumentParser(description='Test realtime note effect')
            parser.add_argument('command', help='Command name')
            parser.add_argument('--hostname', help='WLED hostname or IP address')
            parser.add_argument('--segments', type=int, default=5,
                              help='Number of segments (default: 5)')
            parser.add_argument('--leds-per-segment', type=int, default=16,
                              help='LEDs per segment (default: 16)')

        # Parse again with the command-specific arguments
        args = parser.parse_args()

        # Execute the command
        return commands[args.command](args)
    else:
        logger.error(f"Unknown command: {args.command}")
        show_help()
        return 1

if __name__ == '__main__':
    main()