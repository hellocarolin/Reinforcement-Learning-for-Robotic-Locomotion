# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time
import PCA9685
import pigpio

# http://abyz.me.uk/rpi/pigpio/examples.html

class ServoMotion:
    def __init__(self):
        pi = pigpio.pi(host='192.168.0.10', port=8888)

        if not pi.connected:
            print('connection failed')
            exit(0)

        self.pwm = PCA9685.PWM(pi) # defaults to bus 1, address 0x40

        self.pwm.set_frequency(60) # suitable for servos

        self.servo_min = 2050
        self.servo_mid = 1525
        self.servo_max = 1000

    def set_pwm(self, channel, position):
        self.pwm.set_pulse_width(channel, position)


    def moveHipJointForward(self, leg):
        if leg == 1:
            self.set_pwm(0, self.servo_max)
        elif leg == 2:
            self.set_pwm(4, self.servo_max)
        elif leg == 3:
            self.set_pwm(8, self.servo_min)
        elif leg == 4:
            self.set_pwm(12, self.servo_min)


    def moveHipJointMid(self, leg):
        if leg == 1:
            self.set_pwm(0, self.servo_mid)
        elif leg == 2:
            self.set_pwm(4, self.servo_mid)
        elif leg == 3:
            self.set_pwm(8, self.servo_mid)
        elif leg == 4:
            self.set_pwm(12, self.servo_mid)
        

    def moveHipJointBackwards(self, leg):
        if leg == 1:
            self.set_pwm(0, self.servo_min)
        elif leg == 2:
            self.set_pwm(4, self.servo_min)
        elif leg == 3:
            self.set_pwm(8, self.servo_max)
        elif leg == 4:
            self.set_pwm(12, self.servo_max)


    def moveUpperLegUp(self, leg):
        if leg == 1:
            self.set_pwm(1, self.servo_min)
        elif leg == 2:
            self.set_pwm(5, self.servo_max)
        elif leg == 3:
            self.set_pwm(9, self.servo_min)
        elif leg == 4:
            self.set_pwm(13, self.servo_max)


    def moveUpperLegMid(self, leg):
        if leg == 1:
            self.set_pwm(1, self.servo_mid)
        elif leg == 2:
            self.set_pwm(5, self.servo_mid)
        elif leg == 3:
            self.set_pwm(9, self.servo_mid)
        elif leg == 4:
            self.set_pwm(13, self.servo_mid)

    def moveUpperLegDown(self, leg):
        if leg == 1:
            self.set_pwm(1, self.servo_max)
        elif leg == 2:
            self.set_pwm(5, self.servo_min)
        elif leg == 3:
            self.set_pwm(9, self.servo_max)
        elif leg == 4:
            self.set_pwm(13, self.servo_min)


    def moveLowerLegUp(self, leg):
        if leg == 1:
            self.set_pwm(2, self.servo_max)
        elif leg == 2:
            self.set_pwm(6, self.servo_min)
        elif leg == 3:
            self.set_pwm(10, self.servo_max)
        elif leg == 4:
            self.set_pwm(14, self.servo_min)

    def moveLowerLegMid(self, leg):
        if leg == 1:
            self.set_pwm(2, self.servo_mid)
        elif leg == 2:
            self.set_pwm(6, self.servo_mid)
        elif leg == 3:
            self.set_pwm(10, self.servo_mid)
        elif leg == 4:
            self.set_pwm(14, self.servo_mid)

    def moveLowerLegDown(self, leg):
        if leg == 1:
            self.set_pwm(2, self.servo_min)
        elif leg == 2:
            self.set_pwm(6, self.servo_max)
        elif leg == 3:
            self.set_pwm(10, self.servo_min)
        elif leg == 4:
            self.set_pwm(14, self.servo_max)
