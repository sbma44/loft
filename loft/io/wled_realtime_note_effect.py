import time
import math
import colorsys
import multiprocessing
import socket
import statistics
import os
import random
import sys
import termios
import tty
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def color_animation(hue, progress, peak_position=0.2):
    """
    Generate an RGB color for an animation that transitions from white to a fully saturated color to black.

    Args:
        hue (float): The hue value (0.0-1.0) for the color
        progress (float): The progress through the animation (0.0-1.0)
        peak_position (float): The position in the animation where the fully saturated color appears (0.0-1.0)

    Returns:
        tuple: RGB values as integers (0-255)

    The animation follows this sequence:
    - At progress=0.0: Full white (255, 255, 255)
    - At progress=peak_position: Fully saturated, fully bright version of the hue
    - At progress=1.0: Full black (0, 0, 0)
    """
    if progress <= 0:
        # Start with white
        return (255, 255, 255)
    elif progress >= 1:
        # End with black
        return (0, 0, 0)

    # Normalize the peak position to ensure it's between 0 and 1
    peak_position = max(0.01, min(0.99, peak_position))

    if progress < peak_position:
        # Phase 1: White to fully saturated color
        # Calculate how far we are in this phase (0.0 to 1.0)
        phase_progress = progress / peak_position

        # Convert the target hue to RGB
        target_r, target_g, target_b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]

        # Interpolate from white (255, 255, 255) to the target color
        r = int(255 - (phase_progress * (255 - target_r)))
        g = int(255 - (phase_progress * (255 - target_g)))
        b = int(255 - (phase_progress * (255 - target_b)))

        return (r, g, b)
    else:
        # Phase 2: Fully saturated color to black
        # Calculate how far we are in this phase (0.0 to 1.0)
        phase_progress = (progress - peak_position) / (1 - peak_position)

        # Convert the target hue to RGB
        target_r, target_g, target_b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]

        # Interpolate from the target color to black (0, 0, 0)
        r = int(target_r * (1 - phase_progress))
        g = int(target_g * (1 - phase_progress))
        b = int(target_b * (1 - phase_progress))

        return (r, g, b)

class MultiByteArray:
    def __init__(self, num_leds, segment_size):
        self.segment_size = segment_size
        self.num_leds = num_leds
        self.MAX_UDP_PAYLOAD = 1300  # Maximum UDP payload size in bytes

        # Create a single bytearray for all LEDs (3 bytes per LED for RGB)
        self.data = bytearray(3 * self.num_leds)

        # Calculate how many LEDs can fit in each UDP packet
        # Each LED uses 3 bytes (RGB), and we need 4 bytes for the header
        self.leds_per_packet = (self.MAX_UDP_PAYLOAD - 4) // 3

        # Calculate how many packets we'll need
        self.num_packets = math.ceil(self.num_leds / self.leds_per_packet)

    def __iter__(self):
        # Yield slices of the bytearray with appropriate headers
        for packet_index in range(self.num_packets):
            # Calculate the starting LED index for this packet
            start_led = packet_index * self.leds_per_packet

            # Calculate how many LEDs to include in this packet
            leds_in_packet = min(self.leds_per_packet, self.num_leds - start_led)

            # Create the 4-byte header
            # Byte 0: 4 (DRGB protocol identifier)
            # Byte 1: 2 (protocol version)
            # Bytes 2-3: Starting LED index (high byte, low byte)
            start_index_high_byte = (start_led >> 8) & 0xFF
            start_index_low_byte = start_led & 0xFF
            header = bytearray([4, 2, start_index_high_byte, start_index_low_byte])

            # Calculate byte indices in the data array
            start_byte = start_led * 3
            end_byte = start_byte + (leds_in_packet * 3)

            # Create packet with header + data slice
            packet = bytearray(header)
            packet.extend(self.data[start_byte:end_byte])

            yield packet

    def __getitem__(self, index):
        # Handle single index
        if isinstance(index, int):
            if index < 0 or index >= self.num_leds:
                raise IndexError("LED index out of range")

            # Calculate position in bytearray (3 bytes per LED)
            pos = index * 3

            # Return RGB values as tuple
            return (
                self.data[pos],
                self.data[pos + 1],
                self.data[pos + 2]
            )

        # Handle slice
        elif isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.num_leds
            step = index.step if index.step is not None else 1

            # Create a list of RGB tuples for the slice
            result = []
            for i in range(start, stop, step):
                if i < 0 or i >= self.num_leds:
                    raise IndexError("LED index out of range")
                pos = i * 3
                result.append((
                    self.data[pos],
                    self.data[pos + 1],
                    self.data[pos + 2]
                ))
            return result

        else:
            raise TypeError("Index must be int or slice")

    def __setitem__(self, index, value):
        # Handle slice
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.num_leds
            step = index.step if index.step is not None else 1

            indices = range(start, stop, step)

            # Check if value is a single RGB tuple to be applied to all LEDs in the slice
            if isinstance(value, (tuple, list)) and len(value) == 3 and all(isinstance(x, int) for x in value):
                for i in indices:
                    if i < 0 or i >= self.num_leds:
                        raise IndexError("LED index out of range")
                    pos = i * 3
                    self.data[pos] = value[0]
                    self.data[pos + 1] = value[1]
                    self.data[pos + 2] = value[2]
                return

            # Otherwise, value should be a list of RGB tuples
            if not isinstance(value, (tuple, list)):
                raise ValueError("Value must be RGB tuple or list of RGB tuples")

            if len(value) != len(indices):
                raise ValueError(f"Value list length ({len(value)}) must match slice length ({len(indices)})")

            for i, v in zip(indices, value):
                if not isinstance(v, (tuple, list)) or len(v) != 3:
                    raise ValueError("Each value must be RGB tuple/list of 3 elements")
                if i < 0 or i >= self.num_leds:
                    raise IndexError("LED index out of range")

                pos = i * 3
                self.data[pos] = v[0]
                self.data[pos + 1] = v[1]
                self.data[pos + 2] = v[2]
            return

        # Handle single index
        if isinstance(index, int):
            if index < 0 or index >= self.num_leds:
                raise IndexError("LED index out of range")

            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ValueError("Value must be RGB tuple/list of 3 elements")

            pos = index * 3
            self.data[pos] = value[0]
            self.data[pos + 1] = value[1]
            self.data[pos + 2] = value[2]
        else:
            raise TypeError("Index must be int or slice")

    def set_segment(self, segment_index, color):
        """Set all LEDs in a segment to the same color"""
        if segment_index < 0 or segment_index >= (self.num_leds // self.segment_size):
            raise IndexError("Segment index out of range")

        start = segment_index * self.segment_size
        stop = min((segment_index + 1) * self.segment_size, self.num_leds)

        # Set all LEDs in the segment to the same color
        for i in range(start, stop):
            pos = i * 3
            self.data[pos] = color[0]
            self.data[pos + 1] = color[1]
            self.data[pos + 2] = color[2]

class WLEDRealtimeNoteEffect:
    # Default values for configuration (used if not in .env)
    BLACK = (0, 0, 0)
    MAX_HISTORY_SIZE = 1000  # Maximum number of FPS samples to keep

    def __init__(self, hostname, udp_port, hues, leds_per_segment, max_fps=None):
        """
        Initialize the realtime note effect.

        Args:
            hostname: The hostname or IP address of the WLED device
            hues: List of hue values (0.0-1.0) for each segment
            leds_per_segment: Number of LEDs per segment
            max_fps: Maximum frames per second (overrides env value if provided)
        """
        # Load configuration from environment variables
        self.udp_port = udp_port
        self.note_decay_s = float(os.getenv('NOTE_DECAY_S', '3.0'))

        # Allow max_fps to be overridden by parameter, otherwise use env value
        if max_fps is None:
            self.max_fps = int(os.getenv('MAX_FPS', '60'))
        else:
            self.max_fps = max_fps

        self.frame_time = 1.0 / self.max_fps  # Time per frame in seconds

        # resolve hostname
        self.hostname = hostname
        try:
            info = socket.getaddrinfo(self.hostname, None, socket.AF_INET)
            self.ip_address = info[0][4][0] if info else None
        except socket.gaierror:
            self.ip_address = None
            print(f"Could not resolve hostname: {hostname}")

        # Validate hues
        for hue in hues:
            if not 0 <= hue <= 1.0:
                raise ValueError(f"Hue value {hue} is outside the valid range of 0.0-1.0")

        self.hues = hues
        self.num_segments = len(self.hues)
        self.leds_per_segment = leds_per_segment
        self.num_leds = self.num_segments * self.leds_per_segment

        # Create a pipe for bidirectional communication
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._loop, args=(self.child_conn,))
        self._process.daemon = True
        self._running = False
        self.avg_fps = 0

    def start(self):
        """Start the effect processing in a separate process"""
        if not self._running:
            self._process.start()
            self._running = True

    def wait(self):
        """Wait for the process to complete (blocking)"""
        if self._running:
            self._process.join()

    def stop(self):
        """Stop the effect processing and report average FPS"""
        if self._running:
            # Request FPS stats before stopping
            self.parent_conn.send(('GET_FPS_STATS', None))

            # Wait for response with timeout
            if self.parent_conn.poll(1.0):
                response = self.parent_conn.recv()
                if response[0] == 'FPS_STATS':
                    fps_stats = response[1]
                    if fps_stats['active_frames'] > 0:
                        print(f"\nFPS Statistics:")
                        print(f"  Average FPS: {fps_stats['avg_fps']:.2f}")
                        print(f"  Min FPS: {fps_stats['min_fps']:.2f}")
                        print(f"  Max FPS: {fps_stats['max_fps']:.2f}")
                        print(f"  Active frames: {fps_stats['active_frames']}")
                        print(f"  Total frames: {fps_stats['total_frames']}")
                        self.avg_fps = fps_stats['avg_fps']

            # Send stop signal
            self.parent_conn.send(('STATE', 'STOP'))
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
            self._running = False

    def color_animation(self, hue, progress, peak_position=0.2):
        """
        Generate an RGB color for an animation that transitions from white to a fully saturated color to black.

        Args:
            hue (float): The hue value (0.0-1.0) for the color
            progress (float): The progress through the animation (0.0-1.0)
            peak_position (float): The position in the animation where the fully saturated color appears (0.0-1.0)

        Returns:
            tuple: RGB values as integers (0-255)

        The animation follows this sequence:
        - At progress=0.0: Full white (255, 255, 255)
        - At progress=peak_position: Fully saturated, fully bright version of the hue
        - At progress=1.0: Full black (0, 0, 0)
        """
        if progress <= 0:
            # Start with white
            return (255, 255, 255)
        elif progress >= 1:
            # End with black
            return (0, 0, 0)

        # Normalize the peak position to ensure it's between 0 and 1
        peak_position = max(0.01, min(0.99, peak_position))

        if progress < peak_position:
            # Phase 1: White to fully saturated color
            # Calculate how far we are in this phase (0.0 to 1.0)
            phase_progress = progress / peak_position

            # Convert the target hue to RGB
            target_r, target_g, target_b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]

            # Interpolate from white (255, 255, 255) to the target color
            r = int(255 - (phase_progress * (255 - target_r)))
            g = int(255 - (phase_progress * (255 - target_g)))
            b = int(255 - (phase_progress * (255 - target_b)))

            return (r, g, b)
        else:
            # Phase 2: Fully saturated color to black
            # Calculate how far we are in this phase (0.0 to 1.0)
            phase_progress = (progress - peak_position) / (1 - peak_position)

            # Convert the target hue to RGB
            target_r, target_g, target_b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]

            # Interpolate from the target color to black (0, 0, 0)
            r = int(target_r * (1 - phase_progress))
            g = int(target_g * (1 - phase_progress))
            b = int(target_b * (1 - phase_progress))

            return (r, g, b)

    def _loop(self, child_conn):
        # set up socket
        if not self.ip_address:
            print("No valid IP address, exiting process")
            return

        self.clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clientSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        self.clientSock.setblocking(False)  # Non-blocking mode

        self.state = 'SLEEP'

        self.note_onsets = [-1] * self.num_segments
        self.segment_colors = [self.BLACK] * self.num_segments
        self.ba = MultiByteArray(self.num_leds, self.leds_per_segment)

        # FPS tracking variables using deque with max size
        fps_history = deque(maxlen=self.MAX_HISTORY_SIZE)
        active_fps_history = deque(maxlen=self.MAX_HISTORY_SIZE)
        total_frames = 0
        active_frames = 0

        loop_i = 0
        last_frame_time = time.time()

        while True:
            frame_start_time = time.time()
            is_active_frame = False

            if self.state == 'SLEEP':
                time.sleep(0.01)
            elif self.state == 'STOP':
                break
            else:
                is_active_frame = True
                # look at note onset times, update colors as necessary
                t = time.time()
                for i in range(len(self.note_onsets)):
                    if self.note_onsets[i] < 0:
                        continue
                    else:
                        delta = t - self.note_onsets[i]
                        if delta > self.note_decay_s:
                            self.note_onsets[i] = -1
                            self.ba.set_segment(i, self.BLACK)
                        else:
                            delta_pct = (delta / self.note_decay_s)
                            c = self.color_animation(self.hues[i], delta_pct, peak_position=0.3)
                            self.ba.set_segment(i, c)

                # send bytearray(s)
                try:
                    for msg in self.ba:
                        self.clientSock.sendto(msg, (self.ip_address, self.udp_port))
                except (socket.error, BlockingIOError) as e:
                    print(f"Socket error: {e}")

            # check for messages
            if child_conn.poll():
                msg = child_conn.recv()
                # handle message
                if msg[0] == 'STATE':
                    self.state = msg[1]
                elif msg[0] == 'NOTE':
                    self.note_onsets[msg[1]] = time.time()
                elif msg[0] == 'FPS':
                    self.max_fps = msg[1]
                    self.frame_time = 1.0 / self.max_fps
                elif msg[0] == 'GET_FPS_STATS':
                    # Calculate and send FPS statistics
                    stats = {
                        'avg_fps': statistics.mean(active_fps_history) if active_fps_history else 0,
                        'min_fps': min(active_fps_history) if active_fps_history else 0,
                        'max_fps': max(active_fps_history) if active_fps_history else 0,
                        'active_frames': active_frames,
                        'total_frames': total_frames
                    }
                    child_conn.send(('FPS_STATS', stats))

            # Calculate how long this frame took
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time

            # Sleep to maintain the desired frame rate
            sleep_time = max(0, self.frame_time - frame_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Calculate actual FPS for this frame
            actual_frame_time = time.time() - frame_start_time
            actual_fps = 1.0 / actual_frame_time if actual_frame_time > 0 else 0

            # Track FPS statistics
            fps_history.append(actual_fps)
            if is_active_frame:
                active_fps_history.append(actual_fps)
                active_frames += 1
            total_frames += 1

            loop_i += 1

    def send_note(self, note_index):
        """Send a note event to the effect process"""
        if self._running and 0 <= note_index < self.num_segments:
            self.parent_conn.send(('NOTE', note_index))

    def set_state(self, state):
        """Set the state of the effect process (e.g., 'ACTIVE', 'SLEEP')"""
        if self._running:
            self.parent_conn.send(('STATE', state))

    def set_max_fps(self, max_fps):
        """Update the maximum frames per second"""
        self.max_fps = max_fps
        self.frame_time = 1.0 / self.max_fps
        self.parent_conn.send(('FPS', self.max_fps))

def getch():
    """Get a single character from the user without requiring Enter to be pressed"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Example usage
if __name__ == "__main__":
    # Example parameters
    hostname = os.getenv('WLED_HOSTNAME', 'wled-lamp.local')
    udp_port = int(os.getenv('WLED_UDP_PORT', '21324'))
    hues = [0.0, 0.2, 0.4, 0.6, 0.8]  # Evenly spaced hues
    leds_per_segment = 16

    effect = RealtimeNoteEffect(hostname, udp_port, hues, leds_per_segment)
    print(f"UDP Port: {effect.udp_port}")
    print(f"Note Decay: {effect.note_decay_s}s")
    print(f"Max FPS: {effect.max_fps}")

    effect.start()

    # Set to active state
    effect.set_state('ACTIVE')

    print("\nInteractive Note Tester")
    print("======================")
    print("Press SPACE to trigger a random note")
    print("Press any other key to exit")

    try:
        note_idx = 0
        while True:
            note_idx = (note_idx + 1) % len(hues)

            print(f"Triggering note {note_idx} (hue: {hues[note_idx]:.2f})")
            effect.send_note(note_idx)

            # Wait for key press
            print("Press SPACE for another note or any other key to exit...")
            key = getch()

            # Exit if not space
            if key != ' ':
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        effect.stop()