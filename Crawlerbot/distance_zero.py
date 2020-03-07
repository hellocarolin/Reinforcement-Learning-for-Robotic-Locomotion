import time
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import DistanceSensor
import statistics
import keyboard

# https://gpiozero.readthedocs.io/en/stable/remote_gpio.html#remote-gpio-usage
# https://gpiozero.readthedocs.io/en/stable/api_input.html

class Eyes:
    def __init__(self):

        self.remote_factory = PiGPIOFactory(host='192.168.0.10')

        self.sensor = DistanceSensor(echo=24, trigger=18, pin_factory=self.remote_factory, max_distance=3.0)

    def medianDistance(self):
        distanceArray = []
        for i in range(5):
            distanceArray.append(round(self.sensor.distance * 100, 2))
            print('Distance: ', distanceArray[i])
            time.sleep(.1)
        median = round(statistics.median(distanceArray))
        print(median)
        return median