
import serial
arduino=serial.Serial(port='COM3',baudrate=9600)

while(True):
	state = input()
	arduino.write(bytes(state, 'utf-8'))