from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import keras
import numpy as np
import random
from collections import deque
import pandas as pd
from itertools import product
import time
import threading
import readchar
from sklearn import preprocessing

#https://www.novatec-gmbh.de/en/blog/deep-q-networks/

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.actions = ['0_','1_','2_','_0','_1','_2']
        self.action_size = len(self.actions)
        self.states = np.array(['00','01','11',
                                '10','02','22',
                                '20','12','21'])
        
        self.statesWithLastState = list(product(self.states, repeat=2))
        self.q_matrix = np.zeros((81,6))
        self.q_matrixDF = pd.DataFrame(self.q_matrix, columns=self.actions, index=self.statesWithLastState)
        self.possibleStates = []

        for i in range(len(self.statesWithLastState)):
            if self.statesWithLastState[i][0][0] == self.statesWithLastState[i][1][0] or self.statesWithLastState[i][0][1] == self.statesWithLastState[i][1][1]:
                self.possibleStates.append(self.statesWithLastState[i])
        self.state_size = len(self.possibleStates)

        # removing impossible states from q matrix
        self.q_matrix = np.zeros((45,6))
        self.q_matrixDF = pd.DataFrame(self.q_matrix, columns=self.actions, index=self.possibleStates)

        # 1: decay=0.95, alpha=0.1, gamma=0.9, rewards from q learning
        # 2: same as 1, but new rewards

        self.episodes = 50 #50 #100 
        self.max_episode_steps = 20 #10 #6
        self.epsilon = 1 #1
        self.epsilon_decay = 0.95 #0.95 #0.975 #0.99
        self.epsilon_min = 0.01
        self.gamma = 0.99 #0.99 #0.95
        self.alpha = 0.001  #0.01 #0.001
        self.alpha_decay = 0.001
        self.batch_size = 64
        self.path = "/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/"

        self.model = self._build_model()
        # fixed targets
        self.target_model = self._build_model()
        self.update_target_model()
        self.tau = 0.1

        # adjustment for double DQN
        #self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(2,), activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                    optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    #https://medium.com/@leosimmons/double-dqn-implementation-to-solve-openai-gyms-cartpole-v-0-df554cd0614d
    def soft_update(self):
        q_network_theta = self.model.get_weights()
        target_network_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1-self.tau) + q_weight * self.tau
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_network_theta)

    def act(self, state):
        if (np.random.random() <= self.epsilon):
            print(f"random action: ")
            return random.choice(self.actions)
        state = np.array([state])
        prediction = self.model.predict(state)[0]
        print(f"prediction for next state: {prediction}")
        max_action = np.argmax(prediction)
        print(f"index of max action: {max_action}")
        print(f"predicted action: ")
        return self.actions[max_action]

    def remember(self, state, action, reward, next_state, done):      
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size, paused):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            state = np.array([state])
            next_state = np.array([next_state])
            y_target = self.model.predict(state)[0] # normal dqn
            #y_target = self.target_model.predict(state)[0] # for double dqn/fixed targets
            y_target = DQNAgent.normalizeArray(self, y_target)[0]
            y_nextState = self.model.predict(next_state)[0] #dqn
            #y_target = self.target_model.predict(state) # for double dqn/fixed targets
            y_nextState = DQNAgent.normalizeArray(self, y_nextState)[0]
            y_target[np.where(np.array(self.actions)==action)[0][0]]= reward if done else reward + self.gamma * np.max(y_nextState) #normal dqn
            #y_target[np.where(np.array(self.actions)==action)[0][0]]= reward if done else reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(y_nextState)] #double dqn/fixed targets
            
            x_batch.append(state[0])
            y_batch.append(y_target)

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(np.array(x_batch)), verbose=0)


    def getNextState(self, state, action):
        if action.startswith('_'):
            temp = list(state)
            temp[1] = temp[0] 
            temp[0] = temp[0][0] + action[1]
            nextState = tuple(temp)
        else:
            temp = list(state)
            temp[1] = temp[0]
            temp[0] = action[0] + temp[0][1]
            nextState = tuple(temp)
        return nextState

    def getNextStateWithoutLast(self, df, state, action):
        if action.startswith('_'):
            nextState = state[0] + action[1]
            print(nextState)
        else:
            nextState = action[0] + state[1]
            print(nextState)
        return df[nextState:nextState].index

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
    
    def normalizeArray(self, arrayToNormalize):
        arrayToNormalize = np.reshape(arrayToNormalize, (-1,1))
        scaler = preprocessing.MinMaxScaler([-1,1])
        scaler.fit(arrayToNormalize)
        normalizedArray = scaler.transform(arrayToNormalize)
        normalizedArray = np.reshape(normalizedArray, (1,-1))
        return normalizedArray