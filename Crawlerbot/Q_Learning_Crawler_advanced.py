# import files for servo motion and distance sensor
import servos_zero
import distance_zero
import QLearning_advanced

# import libraries
import numpy as np
import random 
import pandas as pd
import time
import h5py

if __name__ == '__main__':

    servos = servos_zero.ServoMotion()
    eyes = distance_zero.Eyes()
    q = QLearning_advanced.QLearning_advanced(servos)

    leg = 1
    episodes = 50
    maxSteps = 20

    rewards = []

    #randomStartingState = q.q_matrixDF.sample(n=1)
    #state = randomStartingState

    # set random starting state
    #q.setStartingState(state.index.values[0][0], leg)

    for i in range(episodes):
        totalReward = 0

        print(f"episode number: {i}")

        time.sleep(3)

        randomStartingState = q.q_matrixDF.sample(n=1)
        state = randomStartingState

        # set random starting state
        q.setStartingState(state.index.values[0][0], leg)

        for step in range(maxSteps):
            time.sleep(1)
            print(f"step number: {step}")
            
            print(f"state: \n {state}")
            #currentStateLabel = state.index.values[0] # get label of current State

            print("starting distance:")
            startingDistance = eyes.medianDistance() # get starting distance

            if random.uniform(0, 1) > q.epsilon:
                action = q.getBestActionFromQTable(state)
                print(f"best action from q table action: {action}")
            else:
                action = q.getValidRandomAction(state)
                print(f"valid random action: {action}")

            q.doAction(action, leg)

            time.sleep(1)

            print("new distance:")
            newDistance = eyes.medianDistance() # measure new distance
            
            reward = q.getDistanceReward(startingDistance, newDistance) # get reward

            print(f"reward: {reward}")

            nextState = q.getNextState(state, action)
            print(f"next state: {nextState}")

            nextStateRow = q.q_matrixDF.loc[[nextState],:]
            
    
            ### Q_new(s_t, a_t) = (1-alpha)*Q(s_t, a_t)+ alpha*(reward+gamma*max(Q(s_t+1, a_t))
                
            existingQ = q.q_matrixDF.loc[[state.index.values[0]], [action]]

            
            print(f"q.getBestNextValueFromQTable(nextStateRow): \n {q.getBestNextValueFromQTable(nextStateRow)}")

            q.q_matrixDF.loc[[state.index.values[0]], [action]] = ((1-q.alpha) * existingQ) + (q.alpha * (reward + (q.gamma * q.getBestNextValueFromQTable(nextStateRow))))

            print(f"q matrix: \n {q.q_matrixDF}")

            state = nextStateRow #set new state as current state
            

            totalReward = totalReward + reward
            print(f"total reward: \n {totalReward}")

            #if totalReward >= 10:
             #   break

            #time.sleep(0.2)

        q.q_matrixDF.to_csv(r'/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/q_matrix_advanced_v7.csv')

        q.q_matrixDF.to_hdf('q_matrix_advanced_v7.h5', key='q_matrixDF_advanced_v7', mode='w')

        rewards.append(totalReward)

        pd.DataFrame(rewards).to_csv(r'./rewards_Q_Learning_advanced_v7.csv')

        if q.epsilon >= q.epsilon_min:
            q.epsilon *= q.epsilon_decay
            print(f"epsilon is: {q.epsilon}")

        
