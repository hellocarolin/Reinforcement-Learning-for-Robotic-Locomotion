# import files for servo motion and distance sensor
import servos_zero
import distance_zero
import Deep_Q_Learning
import QLearning_advanced
from pynput import keyboard
from pynput.keyboard import Key, Listener
import threading


# import libraries
import numpy as np
import random 
import pandas as pd
import time
import h5py

#https://keon.io/deep-q-learning/

if __name__ == '__main__':
    servos = servos_zero.ServoMotion()
    eyes = distance_zero.Eyes()
    dqn = Deep_Q_Learning.DQNAgent()
    q = QLearning_advanced.QLearning_advanced(servos)

    dqn.model.load_weights('./dqn_crawler_2.h5')

    leg = 1

    rewards = []

    dqn.epsilon = 0.01
    
    randomStartingState = dqn.q_matrixDF.sample().index.values[0]
 
    state = randomStartingState
   
    q.setStartingState(state[0], leg)
    

    while True:
        print(f"state: {state}") 

        action = dqn.act(state)

        q.doAction(action, leg)
        print(f"action: {action}")

        time.sleep(1)

        nextState = dqn.getNextState(state, action)

        print(f"next state: {nextState[0]}")

        state = nextState

        time.sleep(1)

       