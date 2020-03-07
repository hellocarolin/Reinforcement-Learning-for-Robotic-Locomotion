# import files for servo motion and distance sensor
# import libraries
import numpy as np
import random 
import pandas as pd
from itertools import product


class QLearning_advanced:
    
    def __init__(self, servos):
    ### init matrices and parameters ###
        self.servos = servos

        # Q-matrix
        self.q_matrix = np.zeros((81,6))
        self.actions = ['0_','1_','2_','_0','_1','_2']
        self.states = np.array(['00','01','11',
                        '10','02','22',
                        '20','12','21'])

        self.statesWithLastState = list(product(self.states, repeat=2))
        self.q_matrixDF = pd.DataFrame(self.q_matrix, columns=self.actions, index=self.statesWithLastState)
        self.q_matrixDF

        self.possibleStates = []

        for i in range(len(self.statesWithLastState)):
            if self.statesWithLastState[i][0][0] == self.statesWithLastState[i][1][0] or self.statesWithLastState[i][0][1] == self.statesWithLastState[i][1][1]:
                self.possibleStates.append(self.statesWithLastState[i])

        # removing impossible states from q matrix
        self.q_matrix = np.zeros((45,6))
        self.q_matrixDF = pd.DataFrame(self.q_matrix, columns=self.actions, index=self.possibleStates)



        # Set value for the discount factor gamma: how much importance is given to future rewards
        # v3 with gamma=0.8 and alpha=0.2, decay 0.9
        # v2 with gamma = 0.9 and alpha 0.1, decay 0.9
        # original/ v4/v5 gamma =0.8 and alpha 0.65 (epsilon decay 0.9/ 0.95)
        # v6 alpha=0.5, forward movement but including backwards 
        # v7 like v2 but decay 0.95
        
        self.gamma = 0.9 #0.8

        # learning rate
        self.alpha = 0.1 #0.65 #0.2 #0.1 

        # decides if Q-value is used todetermine the action or take a random sample of the action space
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95 #0.9

    ### methods for Q-Learning algorithm ###

    # get reward for movement
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

    # get random action
    def getValidRandomAction(self, state):
        randomAction = state.sample(n=1, axis=1)
        print(f"random action:{randomAction.columns.values[0]}")
        return randomAction.columns.values[0]

    # get the next state for taken action
    def getNextState(self, state, action):
        if action.startswith('_'):
            temp = list(state.index[0])
            temp[1] = temp[0] 
            temp[0] = temp[0][0] + action[1]
            nextState = tuple(temp)
        else:
            temp = list(state.index[0])
            temp[1] = temp[0]
            temp[0] = action[0] + temp[0][1]
            nextState = tuple(temp)
        return nextState

    # method to get all possible actions (not None) from a state row
    def getAllPossibleActions(self,stateRow):
        allPossibleActions = []
        for i in range(len(stateRow.columns)):
          allPossibleActions.append(stateRow.columns.values[i])
        return allPossibleActions

    def setStartingState(self,state,leg):
        if state == '00': 
            print('00')
            self.servos.moveUpperLegDown(leg)
            self.servos.moveLowerLegDown(leg)
        elif state == '01': 
            print('01')
            self.servos.moveUpperLegDown(leg)
            self.servos.moveLowerLegMid(leg)
        elif state == '02': 
            print('02')
            self.servos.moveUpperLegDown(leg)
            self.servos.moveLowerLegUp(leg)
        elif state =='10': 
            print('10')
            self.servos.moveUpperLegMid(leg)
            self.servos.moveLowerLegDown(leg)
        elif state =='11': 
            print('11')
            self.servos.moveUpperLegMid(leg)
            self.servos.moveLowerLegMid(leg) 
        elif state =='12': 
            print('12')
            self.servos.moveUpperLegMid(leg)
            self.servos.moveLowerLegUp(leg)
        elif state =='20': 
            print('20')
            self.servos.moveUpperLegUp(leg)
            self.servos.moveLowerLegDown(leg)
        elif state =='21': 
            print('21')
            self.servos.moveUpperLegUp(leg)
            self.servos.moveLowerLegMid(leg)
        elif state =='22': 
            print('22')  
            self.servos.moveUpperLegUp(leg)
            self.servos.moveLowerLegUp(leg)


    def doAction(self, action, leg):
        if action == "2_":
            self.servos.moveUpperLegUp(leg)
        elif action == "0_":
            self.servos.moveUpperLegDown(leg)
        elif action == "_2":
            self.servos.moveLowerLegUp(leg)
        elif action == "_0":
            self.servos.moveLowerLegDown(leg)
        elif action == "1_":
            self.servos.moveUpperLegMid(leg)
        elif action == "_1":
            self.servos.moveLowerLegMid(leg)

    # function to find best action for current state
    def getBestActionFromQTable(self, state):
        maxValue = state.values[0][0]
        maxAction = state.columns.values[0]
        for i in range(len(state.columns)):
            if state.values[0][i] >= maxValue:
                maxAction = state.columns.values[i]
                maxValue = state.values[0][i]
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
