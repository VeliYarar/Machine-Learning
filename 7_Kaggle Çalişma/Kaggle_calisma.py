# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:38:54 2023

@author: veli1
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Twitter Stock Market Dataset.csv", sep=",")

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)