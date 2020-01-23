# Reinforcement-Learning-for-Quadruped-Locomotion
In this project Reinforcement Learning is researched and looked into from simpler as well as more complex setups. Q-Learning and Deep Q-Learning are applied to a simple crawlerbot and Deep Deterministic Policy Gradient is applied to a complex Quadruped Robot.


## Building a Quadruped Robot
A physical quadruped robot has been built by 3D-printing the body parts using an adjusted template of an existig model for a quadruped robot (https://www.thingiverse.com/thing:2204279). The robot is controlled with a Raspberry Pi 3 and 12 MG90 servo motors, with three motors on each leg having a hip joint, an upper leg joint and a lower joint. For controlling the motors the motor controller board PWM9685 is used. For measuring movements of the robot there are two sensors, an ultrasonic sensor HC-SR04 and an Adafruit BNO055 9-DOF absolute orientation sensor.

## Crawlerbot with Q-Learning and Deep Q-Networks

In the progress of finding a suitable Reinforcement Learning method the quadruped has been transformed to a crawler bot with only two motors being able to perform up to three different movements to achieve a crawling locomotion. For this usecase the Reinforcement Learning methods Q-Learning and Deep Q-Networks have been implemented whereby Q-Learning can outperform the more complex Deep Q-Networks for such a simple setup due to time-consuming tweaking of the hyper-parameters and longer training to approach a perfect locomotion. As the setup f the crawlerbot is quite simple, it can be trained in reality using remote GPIO.

<p align="center">
<img src="https://github.com/hellocarolin/Reinforcement-Learning-for-Quadruped-Locomotion/blob/master/crawlerbot1.jpg" width="240"     height="160" title="Crawlerbot">
</p>

### Run Q-Learning
Run Q_Learning_Crawler_advanced.py to train Q-Learning on the Crawlerbot also needing the servos_zero.py and distance_zero.py for controlling the servos and measuring the distance both remotely so that the transmission of the signals is fast enough. In addition, the file QLearning_advanced.py needs to be imported having all the functions that are used in the algorithm. To test the learned actions run Q_Learning_Crawler_advanced_Test.py and it will execute the learned optimal actions starting from a random starting state.

### Run DQN
Similarly, run DQN_Crawler.py to train DQN on the Crawlerbot also needing the servos_zero.py and distance_zero.py for controlling the servos and measuring the distance. In addition, the file Deep_Q_Learning.py needs to be imported having all the functions that are used in the algorithm. To test the algorithm one can set the parameter epsilon to 0 so that the agent would only exploit, i.e. peform the learned actions and not try random actions anymore.

## Quadruped Robot with Deep Deterministic Policy Gradient

For the complex quadruped robot the Deep Deterministic Policy Gradient is chosen to solve the problem. It is implemented using a solution by YunjaeChoi (https://github.com/YunjaeChoi/Reinforment-Implementation-on-a-Quadruped). This needs to be adjusted for the usecase and implemented with python 2.7, ROS Kinetic, Tensorflow and Gazebo simulation. This way, the training of this model needs to be done in simulation as it very time-consuming. The setup needs to be done as YunjaeChoi suggests. This repository contains a cloned version of that repository with adjustments for this usecase. Especially, the transmission to the servo motors has to be adjusted depending on how they are appended to the legs. This needs to be done in quadruped_model.py. Furthermore, the adjustment of the model of the robot in the simulation is essential as each small difference can make the transfer to the reality different or not as expected. The model can be changed in the file quadruped_model.xacro. When launching gazebo with roslaunch quadruped quadruped_control.launch, one can directly see the changes of size whereas mass is quite difficult to adjust. By following the instructions on the referred repository (https://github.com/YunjaeChoi/Reinforment-Implementation-on-a-Quadruped) on can train the robot in simluation and afterwards project the learned behavior onto the physical robot in the real world. Having the BNO055 absolute orientation sensor, one can even let the quadruped robot move perceiving the real environment.

<p align="center">
<img src="https://github.com/hellocarolin/Reinforcement-Learning-for-Quadruped-Locomotion/blob/master/quadruped1.jpg" width="240"     height="160" title="Quadruped Robot">
</p>

Videos of the resulting locomotion can be found here:
- Q-Learning & DQN for Crawlerbot: link1
- DDPG for Quadruped Robot: link2
