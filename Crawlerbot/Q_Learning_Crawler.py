# import files for servo motion and distance sensor
import servos_zero
import distance_zero
import QLearning

# import libraries
import numpy as np
import random 
import pandas as pd
import time
import h5py

# https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56

### TRAINING ###:

if __name__ == '__main__':

    servos = servos_zero.ServoMotion()
    eyes = distance_zero.Eyes()
    q = QLearning.QLearning(servos)

    leg = 1

    episodes= 20
    maxSteps = 20

    rewards = []

    for i in range(episodes):
        totalReward = 0
        print(f"episode number: {i}")

        randomStartingState = q.rewardDF.sample(n=1)
        startingState = randomStartingState

        # set random starting state
        q.setStartingState(startingState, leg)

        for step in range(maxSteps):
            print(f"step number: {step}")
            time.sleep(1)
            currentState = startingState.index.values[0] # get label of current State
            print(f"starting state: {startingState}")
            print(f"current state: {currentState}")

            print("starting distance")
            startingDistance = eyes.medianDistance() # get starting distance

            if random.uniform(0, 1) > q.epsilon:
                action = q.getBestActionFromQTable(startingState)
                print(f"best action from q table action: {action}")
            else:
                action = q.getValidRandomAction(startingState)
                print(f"valid random action: {action}")

            #validRandomAction = q.getValidRandomAction(startingState) # get valid random actions from state
            #q.doAction(validRandomAction, leg) # execute action with servos

            q.doAction(action, leg)

            time.sleep(1)

            print("new distance")
            newDistance = eyes.medianDistance() # measure new distance
            
            reward = q.getDistanceReward(startingDistance, newDistance) # get reward

            print(f"reward: {reward}")

            print(q.rewardDF.loc[currentState][action])
            #q.rewardDF.loc[validRandomAction.index.values[0]][validRandomAction.columns.values[0]] = reward # put reward in reward table
            q.rewardDF.loc[currentState][action] = reward # put reward in reward table


            print(f"reward table: \n {q.rewardDF}")
            
            #actionHappened = validRandomAction.columns.values[0] # get label of action just executed
            actionHappened = action
            nextState = q.getNextState(currentState, actionHappened)
            print(f"nextState: \n {nextState}")
            nextStateRow = q.q_matrixDF.loc[[nextState],:]
            print(f"nextStateRow: \n {nextStateRow}")
    


            # update q matrix: Q_new(s_t, a_t) = (1-alpha)*Q(s_t, a_t)+ alpha*(reward+gamma*max(Q(s_t+1, a_t))
            
            qNew = q.q_matrixDF.loc[currentState, actionHappened]
            qOld = q.q_matrixDF.loc[currentState, actionHappened]

            print(f"best next value from q table: \n {q.getBestNextValueFromQTable(nextStateRow)}")
            
            qNew = ((1-q.alpha) * qOld) + (q.alpha * (reward + (q.gamma * q.getBestNextValueFromQTable(nextStateRow))))
            
            q.q_matrixDF.loc[currentState, actionHappened] = qNew
            
            print(f"q matrix: \n {q.q_matrixDF}")
            
            startingState = nextStateRow #set new state as current state

            totalReward = totalReward + reward
            print(f"total reward: \n {totalReward}")

        q.q_matrixDF.to_csv(r'/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/q_matrix.csv')
        q.rewardDF.to_csv(r'/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/reward_matrix.csv')

        q.q_matrixDF.to_hdf('q_matrix.h5', key='q_matrixDF', mode='w')
        q.rewardDF.to_hdf('reward_matrix.h5', key='rewardDF', mode='w')
        
        rewards.append(totalReward)

        pd.DataFrame(rewards).to_csv(r'./rewards_Q_Learning.csv')

        if q.epsilon >= q.epsilon_min:
            q.epsilon *= q.epsilon_decay
            print(f"epsilon is: {q.epsilon}")





    


