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
    leg = 1
    rewards = []
    
    for i in range(dqn.episodes):
        totalReward = 0
        totalDistance = 0
        done = False
        randomStartingState = dqn.q_matrixDF.sample().index.values[0]
        state = randomStartingState
        time.sleep(1)

        # set random starting state
        q.setStartingState(state[0], leg)
    
        for step in range(dqn.max_episode_steps):
            print(f"episode number: {i+1}")
            print(f"step number: {step+1}")
            time.sleep(1)
            print(f"state: {state}")
            print("starting distance:")

            startingDistance = eyes.medianDistance()
            action = dqn.act(state)
            q.doAction(action, leg)
            print(f"action: {action}")

            time.sleep(1)

            print("new distance: ")
            newDistance = eyes.medianDistance() # measure new distance
            
            reward = dqn.getDistanceReward(startingDistance, newDistance) # get reward

            print(f"reward: {reward}")

            nextState = dqn.getNextState(state, action)
            print(f"next state: {nextState[0]}")

            totalReward = totalReward + reward
            print(f"total reward: {totalReward}")

            if totalReward >= 10:
                done = True

            dqn.remember(state,action,reward,nextState,done)

            state = nextState

            dqn.replay(dqn.batch_size, pausing)

            # soft update of target network
            dqn.soft_update()

            dqn.model.save_weights('./dqn_double_fixed_crawler_2.h5')

            if done:
                break

        rewards.append(totalReward)

        pd.DataFrame(dqn.memory).to_csv(r'./memory_dqn_double_fixed_2.csv')

        pd.DataFrame(rewards).to_csv(r'./rewards_dqn_double_fixed_2.csv')

        print(f"epsilon: {dqn.epsilon}")
        if dqn.epsilon > dqn.epsilon_min:
            dqn.epsilon *= dqn.epsilon_decay
        
        
       


