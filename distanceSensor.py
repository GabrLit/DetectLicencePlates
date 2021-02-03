import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO_TRIG=4
GPIO_ECHO=18

GPIO.setup(GPIO_TRIG, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():

    GPIO.output(GPIO_TRIG, True)
    time.sleep(0.0001)
    GPIO.output(GPIO_TRIG, False)

    startTime= time.time()
    stopTime= time.time()

    while GPIO.input(GPIO_ECHO)==0:
        startTime= time.time()

    while GPIO.input(GPIO_ECHO)==1:
        stopTime= time.time()


    elapsed= stopTime-startTime
    distance= elapsed*34300/2

    return distance

if __name__== '__main__':
    try:
        while True:
            dist=distance()
            print("Distance: %.1f cm" %dist)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n Stopped by user")
        GPIO.cleanup()
