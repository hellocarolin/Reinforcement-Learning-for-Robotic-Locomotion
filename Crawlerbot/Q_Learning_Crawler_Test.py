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


### TEST ###:
if __name__ == '__main__':
    servos = servos_zero.ServoMotion()
    eyes = distance_zero.Eyes()
    q = QLearning.QLearning(servos)

    q_matrixDF = pd.read_hdf('q_matrix.h5', 'q_matrixDF')

    leg = 1

    randomStartingState = q_matrixDF.sample(n=1)
    startingState = randomStartingState

    # set random starting state
    q.setStartingState(startingState, leg)

    while True:
        currentState = startingState.index.values[0] # get label of current State

        print("get best action from q table...")  
        optimalAction = q.getBestActionFromQTable(startingState)

        print("do action")
        q.doAction(optimalAction, leg) # execute action with servos
    
        actionHappened = optimalAction # get label of action just executed

        print(f"action happened: {actionHappened}") 
        print(f"currentState: {currentState}") 

        nextState = q.getNextState(currentState, actionHappened)
        print(f"nextState: {nextState}")
        nextStateRow = q_matrixDF.loc[[nextState],:]
        print(f"nextStateRow: {nextStateRow}")

        startingState = nextStateRow #set new state as current state

        print("next state set")
        time.sleep(1)
