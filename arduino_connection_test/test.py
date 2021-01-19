import pyfirmata
import time

# connect to board via firmata
board = pyfirmata.Arduino('/dev/cu.usbmodem1411')

# blink 13
while True:
    board.digital[13].write(1)
    time.sleep(1)
    board.digital[13].write(0)
    time.sleep(1)