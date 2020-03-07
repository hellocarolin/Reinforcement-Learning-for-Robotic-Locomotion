# import libraries
import numpy as np
import random 
import pandas as pd

class QLearning:
    
    def __init__(self, servos):
    ### init matrices and parameters ###
        self.servos = servos

        # reward matrix
        self.reward = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        self.rewardDF = pd.DataFrame(self.reward, columns=['U_','D_','_U','_D'], 
                                        index=['UU','UD','DU','DD'])
        # Q-matrix
        #q_matrix = np.zeros((4,4))
        self.q_matrix = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        self.q_matrixDF = pd.DataFrame(self.q_matrix, columns=['U_','D_','_U','_D'], 
                                            index=['UU','UD','DU','DD'])



        # Set value for the discount factor gamma: how much importance is given to future rewards
        self.gamma = 0.9

        # learning rate
        self.alpha = 0.1

        # epsilon
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9

    ### methods for Q-Learning algorithm ###
    
    # get reward for movement - Q-Learning advanced
    def getDistanceReward(self, dist1, dist2):
        distanceReward = 0
        if dist2 < dist1:
            if dist1 - dist2 > 3:
                distanceReward = 10
            else: 
                distanceReward = 5
        elif dist2 == dist1:
            distanceReward = -2
        elif dist1 - dist2 < -1:
            distanceReward = -10
        else:
            distanceReward = -5
        return distanceReward

    def getValidRandomAction(self, state):
        randomAction = state.sample(n=1, axis=1)
        print(randomAction.columns.values[0])
        return randomAction.columns.values[0]

    # get the next state for taken action
    def getNextState(self, state, action):
        if action.startswith('_'):
            nextState = str(state)[0] + str(action)[1]
        else:
            nextState = str(action)[0] + str(state)[1]
        return nextState

    def getAllPossibleActions(self, state):
        allPossibleActions = []
        for i in range(len(state.columns)):
            allPossibleActions.append(state.columns.values[i])
        return allPossibleActions

    def setStartingState(self, startingState, leg):
        stateName = startingState.index.values[0]

        if stateName == "UU":
            self.servos.moveUpperLegUp(leg)
            self.servos.moveLowerLegUp(leg)
        elif stateName == "UD":
            self.servos.moveUpperLegUp(leg)
            self.servos.moveLowerLegDown(leg)
        elif stateName == "DD":
            self.servos.moveUpperLegDown(leg)
            self.servos.moveLowerLegDown(leg)
        elif stateName == "DU":
            self.servos.moveUpperLegDown(leg)
            self.servos.moveLowerLegUp(leg)

    def doAction(self, action, leg):
        if isinstance(action, str):
            actionName = action
        else:
            actionName = action.columns.values[0]

        if actionName == "U_":
            self.servos.moveUpperLegUp(leg)
        elif actionName == "D_":
            self.servos.moveUpperLegDown(leg)
        elif actionName == "_U":
            self.servos.moveLowerLegUp(leg)
        elif actionName == "_D":
            self.servos.moveLowerLegDown(leg)

    def getBestActionFromQTable(self, state):
        maxValue = state.values[0][0]
        #print(f"Max value: {maxValue}")
        maxAction = state.columns.values[0]
        #print(f"Max action: {maxAction}")
        for i in range(len(state.columns)):
            if state.values[0][i] >= maxValue:
                maxAction = state.columns.values[i]
                #print(f"Max action: {maxAction}")
                maxValue = state.values[0][i]
                #print(f"Max value: {maxValue}")
            else:
                continue
        return maxAction

    def getBestNextValueFromQTable(self, state):
        maxValue = state.values[0][0]
        for i in range(len(state.columns)):
            if state.values[0][i] >= maxValue:
                maxValue = state.values[0][i]
            else:
                continue
        return maxValue

            
