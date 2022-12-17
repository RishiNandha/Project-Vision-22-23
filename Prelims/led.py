import serial
arduino=serial.Serial(port='COM3',baudrate=115200, timeout=.1)
while (True):
	x = input()
	arduino.write(bytes(x,'utf-8'))