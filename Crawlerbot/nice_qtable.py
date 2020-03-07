import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv(filepath_or_buffer='/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/q_matrix_advanced_v7.csv', sep=",")

data = pd.DataFrame(data)

round(data, 2).to_csv(r'/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/src/nice_qtable_advanced_v7.csv')