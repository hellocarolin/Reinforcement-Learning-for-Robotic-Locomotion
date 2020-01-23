# Reinforcement-Learning-for-Quadruped-Locomotion

In this project a physical quadruped robot has been built using a Raspberry Pi 3 and 12 MG90 servo motors, with three motors on each leg having a hip join, an upper leg joint and a lower joint. For controlling the motors the motor controller board PWM9685 is used. For measuring movements of the robot there are two sensors, an ultrasonic sensor HC-SR04 and a Adafruit BNO055 9-DOF absolute orientation sensor.

In the progress of finding a suitable Reinforcement Learning method a the quadruped has been transformed to a crawler bot with only two motors being able to perform up to three different movements to achieve a crawling locomotion. For this usecase the Reinforcement Learning methods Q-Learning and Deep Q-Networks have been implemented whereby Q-Learning can outperform the more complex Deep Q-Networks for such a simple setup due to time-consuming tweaking of the hyper-parameters and longer training to approach a perfect locomotion. As the setup f the crawlerbot is quite simple, it can be trained in reality using remote GPIO.

<p align="center">
<img src="https://github.com/hellocarolin/Reinforcement-Learning-for-Quadruped-Locomotion/blob/master/crawlerbot1.jpg" width="240"     height="160" title="Crawlerbot">
</p>

For the complex quadruped robot the Deep Deterministic Policy Gradient is chosen to solve the problem. It is implemented using a solution by YunjaeChoi (https://github.com/YunjaeChoi/Reinforment-Implementation-on-a-Quadruped). This needs to be adjusted for the usecase and implemented with Gazebo and ROS. This way, the training of this model needs to be done in simulation as it very time-consuming.

<p align="center">
<img src="https://github.com/hellocarolin/Reinforcement-Learning-for-Quadruped-Locomotion/blob/master/quadruped1.jpg" width="240"     height="160" title="Quadruped Robot">
</p>

Videos of the resulting locomotion can be found here:
- Q-Learning & DQN for Crawlerbot: link1
- DDPG for Quadruped Robot: link2
