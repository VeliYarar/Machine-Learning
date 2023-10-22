# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:15:36 2023

@author: veli1
"""

import pandas as pd
import matplotlib.pyplot as plt
 

df = pd.read_csv("polynomial+regression.csv", sep=";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")

plt.show()

#linear regression  y0 = b0 + b1*x
#multiple linear regression  y0 = b0 + b1*x1 +b2*x2

#%% linear regression

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression() 

linear_reg.fit(x, y)

#%%

y_head = linear_reg.predict(x)

plt.plot(x, y_head, color = "red")
plt.show()

linear_reg.predict([[10000]])# eger araba 10 milyonsa 871 ile gider sonucu verir, bu yanlıştır
# cünkü datamız linear reg'e uygun değil

#%%
#polynomial linear regression y0 = b0 + b1*x + b2*x^2 + b3*x^3+...

from sklearn.preprocessing import PolynomialFeatures

pol_reg = PolynomialFeatures(degree=4)

x_pol = pol_reg.fit_transform(x)
#%%

linear_reg2 = LinearRegression()

linear_reg2.fit(x_pol, y)
#%%

y_head2 = linear_reg2.predict(x_pol)

plt.plot(x,y_head2, color = "black", label ="poly")
plt.legend()
plt.show()














