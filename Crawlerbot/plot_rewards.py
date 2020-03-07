import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data_Q_Learning_advanced = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/rewards_Q_Learning_advanced_v7.csv', sep=",")
data_Q_Learning_simple = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/rewards_Q_Learning.csv', sep=",")
data_dqn_simple = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/rewards_dqn_2.csv', sep=",")
data_dqn_double = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/rewards_dqn_double_fixed_2.csv', sep=",")

data_advanced = pd.DataFrame(data_Q_Learning_advanced)
data_simple = pd.DataFrame(data_Q_Learning_simple)
data_dqn_simple = pd.DataFrame(data_dqn_simple)
data_dqn_double = pd.DataFrame(data_dqn_double)

data_simple.columns = ['episode', 'total reward']

X_ad = np.array((pd.Series(range(1,51)))).reshape(-1, 1) 
Y_ad = data_advanced.iloc[:, 1].values.reshape(-1, 1)  
X = np.array((pd.Series(range(1,21)))).reshape(-1, 1)   
Y = data_simple.iloc[:, 1].values.reshape(-1, 1)
X_dqn = np.array((pd.Series(range(1,51)))).reshape(-1, 1) 
Y_dqn = data_dqn_simple.iloc[:, 1].values.reshape(-1, 1)  
X_dqn_d = np.array((pd.Series(range(1,51)))).reshape(-1, 1) 
Y_dqn_d = data_dqn_double.iloc[:, 1].values.reshape(-1, 1)

linear_regressor = LinearRegression()  
linear_regressor.fit(X, Y)  
Y_pred = linear_regressor.predict(X)  
linear_regressor.fit(X_ad, Y_ad)  
Y_ad_pred = linear_regressor.predict(X_ad) 


linear_regressor.fit(X_dqn, Y_dqn)  
Y_dqn_pred = linear_regressor.predict(X_dqn)  
linear_regressor.fit(X_dqn_d, Y_dqn_d)  
Y_dqn_d_pred = linear_regressor.predict(X_dqn_d) 
plt.subplots(1, 1, figsize=(10,6))

dqn = plt.scatter(X_dqn,Y_dqn, color='orchid')
plt.plot(X_dqn, Y_dqn_pred, color = 'orchid', label='DQN')
dqn_double = plt.scatter(X_dqn_d,Y_dqn_d, color = 'lightgrey')
plt.plot(X_dqn_d, Y_dqn_d_pred, color = 'lightgrey', label='Double DQN')

q_advanced = plt.scatter(X_ad,Y_ad, color='lightblue')
plt.plot(X_ad, Y_ad_pred, color = 'lightblue', label='Advanced Q-Learning')
q = plt.scatter(X,Y, color = 'lightgreen')
plt.plot(X, Y_pred, color = 'lightgreen', label='Q-Learning')

plt.axis([0, 51, -60, 25])
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)

plt.ylabel('total rewards', fontsize=12)
plt.xlabel('episodes', fontsize=12)


plt.legend([q, q_advanced, dqn, dqn_double], ['Q-Learning', 'Advanced Q-Learning', 'DQN', 'Double DQN'], loc="center right",
           borderaxespad=-15, fontsize=11)        
plt.subplots_adjust(right=0.75)

plt.show()
