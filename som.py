import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Importing the dataset
data = pd.read_csv("Credit_Card_Applications.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Feature Scaling <we will use MinMax Scaler as we saw in EDA Data doesn't follow the bell curve>
mx = MinMaxScaler(feature_range=(0,1))
x = mx.fit_transform(x)

#Training The SOM
som = MiniSom(x=10,y=10,sigma=1.0,learning_rate=0.5,input_len=15)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)

# Visualizing the Resuls
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, j in enumerate(x):
    w = som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding The Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(5,4)],mappings[(5,2)]),axis=0)
frauds = mx.inverse_transform(frauds)
frauds = pd.DataFrame(frauds,columns=data.columns[:-1])
frauds.iloc[:,0].to_csv("frauds.csv",header=False,index=False)