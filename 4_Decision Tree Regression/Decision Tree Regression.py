# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:10:46 2023

@author: veli1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("decision+tree+regression+dataset.csv", sep = ";", header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%  # decicion tree regression

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict([[6.0]])
x_= np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x)
#%% visualize

plt.scatter(x,y,color = "red")
plt.plot(x_,y_head, color = "green")

plt.xlabel("tribun")
plt.ylabel("ucret")

plt.show()