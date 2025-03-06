FROM python:3.11-slim

RUN sudo apt-get update -y

RUN apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    portaudio19-dev

    # necessary? /etc/asound.conf
    # defaults.ctl.card 2
    # defaults.pcm.card 2

# install and run jack audio server
# seems like aubio isn't using jack tho
# RUN sudo apt install jackd2
# RUN jack_control start

RUN sudo apt-get install -y python3 python3-pip python3-dev git
RUN sudo apt install -y python3-gpiozero
RUN sudo pip3 install -y lgpio rpi_ws281x adafruit-circuitpython-neopixel python-dotenv pyserial pyaudio git+https://git.aubio.org/aubio/aubio/
RUN sudo python3 -m pip install --force-reinstall adafruit-blinka
RUN rm -rf /var/lib/apt/lists/*