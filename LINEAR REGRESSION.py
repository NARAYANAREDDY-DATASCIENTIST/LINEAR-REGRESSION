# -*- coding: utf-8 -*-
"""
Spyder Editor

 @ NARAYANA REDDY DATA SCIENTIST  """
 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# READ THE DATA SET
dataset=pd.read_csv('salary_data.csv')

# DIVIDE THE DATASET INTO THE INDEPENDENT AND DEPEDENT VARIABLE
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

# SPLITTING THE DATASET INTO TRAIN AND TEST DATA SET

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=29)

# IMPLEMENT SIMPLE LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
SimpleLineaRregression=LinearRegression()
# FIT THE MODEL
SimpleLineaRregression.fit(x_train,y_train)

# PREDICT THE MODEL
y_predict=SimpleLineaRregression.predict(x_test)

#PLOT THE GRAPH

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,SimpleLineaRregression.predict(x_train))
plt.show()

# MODEL SCORE
from sklearn.metrics import r2_score
modelscore=r2_score(y_test,y_predict)
