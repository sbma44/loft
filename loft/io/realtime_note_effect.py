import time
import math
import colorsys
import multiprocessing
import socket

class MultiByteArray:
    def __init__(self, num_leds, segment_size):
        self.segment_size = segment_size
        self.num_leds = num_leds

        MAX_UDP_SIZE = 1300  # bytes
        required_bytearrays = math.ceil((3 * self.num_leds) / MAX_UDP_SIZE)
        self.bytearrays = [bytearray(round(3 * self.num_leds / required_bytearrays))]
        for i in range(len(self.bytearrays)):
            self.bytearrays[i].extend([0] * (len(self.bytearrays[i])))

        self.bytearray_starts = [0]
        for i in range(len(self.bytearrays) - 1):
            self.bytearray_starts.append(len(self.bytearrays[i]))

    def __iter__(self):
        # Yield each bytearray with 4 byte header prepended
        for (i, ba) in enumerate(self.bytearrays):
            # Create new bytearray with 4 byte header + data
            start_index = int(self.bytearray_starts[i] / 3)  # Convert to int first
            start_index_high_byte = (start_index >> 8) & 0xFF  # Get high byte
            start_index_low_byte = start_index & 0xFF  # Get low byte

            with_header = bytearray([4, 2, start_index_high_byte, start_index_low_byte])
            with_header.extend(ba)
            yield with_header

    def __getitem__(self, index):
        # Find which bytearray contains this index
        byte_index = index * 3  # Each LED uses 3 bytes (RGB)
        array_index = 0
        while array_index < len(self.bytearray_starts) - 1 and byte_index >= self.bytearray_starts[array_index + 1]:
            array_index += 1

        # Get the relative position within that bytearray
        relative_pos = byte_index - self.bytearray_starts[array_index]

        # Return RGB values as tuple
        return (
            self.bytearrays[array_index][relative_pos],
            self.bytearrays[array_index][relative_pos + 1],
            self.bytearrays[array_index][relative_pos + 2]
        )

    def __setitem__(self, index, value):
        # Handle slice
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.num_leds
            step = index.step if index.step is not None else 1

            if not isinstance(value, (tuple, list)):
                raise ValueError("Value must be list of RGB tuples/lists for slice assignment")

            indices = range(start, stop, step)
            if len(value) != len(indices):
                raise ValueError("Value list length must match slice length")

            for i, v in zip(indices, value):
                if not isinstance(v, (tuple, list)) or len(v) != 3:
                    raise ValueError("Each value must be RGB tuple/list of 3 elements")
                self._set_single_item(i, v)
            return

        # Handle single index
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError("Value must be RGB tuple/list of 3 elements")
        self._set_single_item(index, value)

    def _set_single_item(self, index, value):
        # Find which bytearray contains this index
        byte_index = index * 3  # Each LED uses 3 bytes
        array_index = 0
        while array_index < len(self.bytearray_starts) - 1 and byte_index >= self.bytearray_starts[array_index + 1]:
            array_index += 1

        # Get the relative position within that bytearray
        relative_pos = byte_index - self.bytearray_starts[array_index]

        # Set RGB values
        self.bytearrays[array_index][relative_pos] = value[0]
        self.bytearrays[array_index][relative_pos + 1] = value[1]
        self.bytearrays[array_index][relative_pos + 2] = value[2]

    def set_segment(self, segment_index, color):
        segment_colors = [color] * self.segment_size
        start = segment_index * self.segment_size
        stop = (segment_index + 1) * self.segment_size
        self[start:stop] = segment_colors

class RealtimeNoteEffect:
    UDP_PORT_NO = 21324
    CHECK_INTERVAL = 10
    NOTE_DECAY_S = 3.0
    BLACK = (0, 0, 0)
    MAX_FPS = 60  # Maximum frames per second

    def __init__(self, hostname, colors, leds_per_segment, max_fps=MAX_FPS):
        # resolve hostname
        self.hostname = hostname
        try:
            info = socket.getaddrinfo(self.hostname, None, socket.AF_INET)
            self.ip_address = info[0][4][0] if info else None
        except socket.gaierror:
            self.ip_address = None
            print(f"Could not resolve hostname: {hostname}")

        self.colors = colors
        self.num_segments = len(self.colors)
        self.leds_per_segment = leds_per_segment
        self.num_leds = self.num_segments * self.leds_per_segment
        self.max_fps = max_fps
        self.frame_time = 1.0 / self.max_fps  # Time per frame in seconds

        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._loop, args=(self.child_conn,))
        self._process.daemon = True
        self._running = False

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
        """Stop the effect processing"""
        if self._running:
            self.parent_conn.send(('STATE', 'STOP'))
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
            self._running = False

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

        loop_i = 0
        last_frame_time = time.time()

        while True:
            frame_start_time = time.time()

            if self.state == 'SLEEP':
                time.sleep(0.01)
            elif self.state == 'STOP':
                break
            else:
                # look at note onset times, update colors as necessary
                t = time.time()
                for i in range(len(self.note_onsets)):
                    if self.note_onsets[i] < 0:
                        continue
                    else:
                        delta = t - self.note_onsets[i]
                        if delta > self.NOTE_DECAY_S:
                            self.note_onsets[i] = -1
                            self.ba.set_segment(i, self.BLACK)
                        else:
                            delta_pct = 1.0 - (delta / self.NOTE_DECAY_S)
                            c = [int(255 * x) for x in colorsys.hsv_to_rgb(1.0, 1.0, delta_pct)]
                            self.ba.set_segment(i, c)

                # send bytearray(s)
                try:
                    for msg in self.ba:
                        self.clientSock.sendto(msg, (self.ip_address, self.UDP_PORT_NO))
                except (socket.error, BlockingIOError) as e:
                    print(f"Socket error: {e}")

            # check for messages every CHECK_INTERVAL loops
            if (loop_i % self.CHECK_INTERVAL) == 0:
                if child_conn.poll():
                    msg = child_conn.recv()
                    # handle message
                    if msg[0] == 'STATE':
                        self.state = msg[1]
                    elif msg[0] == 'NOTE':
                        self.note_onsets[msg[1]] = time.time()

            # Calculate how long this frame took
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time

            # Sleep to maintain the desired frame rate
            sleep_time = max(0, self.frame_time - frame_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Calculate actual FPS for debugging if needed
            actual_frame_time = time.time() - frame_start_time
            actual_fps = 1.0 / actual_frame_time if actual_frame_time > 0 else 0

            print(actual_fps)

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

# Example usage
if __name__ == "__main__":
    # Example parameters
    hostname = "wled-lamp.local"
    colors = [(255, 0, 0), (0, 255, 0)]  # RGB colors for each segment
    leds_per_segment = 20

    effect = RealtimeNoteEffect(hostname, colors, leds_per_segment)
    effect.start()

    # Set to active state
    effect.set_state('ACTIVE')

    # Simulate some notes
    import random
    try:
        for _ in range(2):
            note_idx = _
            effect.send_note(note_idx)
            time.sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        effect.stop()