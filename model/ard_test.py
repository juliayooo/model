import serial
import time

ser = serial.Serial('/dev/tty.usbmodem101', 9600)  # Update with your detected port
time.sleep(2)  # Wait for the connection to establish

ser.write(b'1')  # Send a test signal to blink once
print("Sent signal to Arduino!")

ser.close()