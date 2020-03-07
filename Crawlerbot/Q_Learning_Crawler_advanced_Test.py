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

### TEST ###:
if __name__ == '__main__':
    servos = servos_zero.ServoMotion()
    eyes = distance_zero.Eyes()
    q = QLearning_advanced.QLearning_advanced(servos)

    #rewardDF = pd.read_hdf('reward_matrix.h5', 'rewardDF')
    q_matrixDF = pd.read_hdf('q_matrix_advanced_v6.h5', 'q_matrixDF_advanced_v6')
    #q_matrixDF = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/q_matrix_advanced.csv', sep=",")
    #q_matrixDF = pd.DataFrame(q_matrixDF)
    #q_matrixDF.to_hdf('q_matrix_advanced_original.h5', key='q_matrixDF_advanced_original', mode='w')
    leg = 1

    randomStartingState = q_matrixDF.sample(n=1)
    state = randomStartingState

    # set random starting state
    q.setStartingState(state.index.values[0][0], leg)

    while True:
        currentState = state.index.values[0] # get label of current State

        print("get best action from q table...")  
        action = q.getBestActionFromQTable(state)

        print("do action")
        q.doAction(action, leg) # execute action with servos

        print(f"action happened: {action}") 
        print(f"current state: {state}") 

        nextState = q.getNextState(state, action)
        print(f"nextState: {nextState}")
        nextStateRow = q_matrixDF.loc[[nextState],:]
        print(f"nextStateRow: {nextStateRow}")

        state = nextStateRow #set new state as current state

        print("next state set!")
        time.sleep(5)
