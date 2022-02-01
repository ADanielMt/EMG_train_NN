c=a# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:10:38 2021

@author: AlejandroDaniel
"""

import pandas as pd
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier   
from pickle import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


names = ['iav1','iav2','iav3','iav4','iav5','iav6','iav7','iav8', 
         'mav1','mav2','mav3','mav4','mav5','mav6','mav7','mav8',
         'sqrt1','sqrt2','sqrt3','sqrt4','sqrt5','sqrt6','sqrt7','sqrt8', 
         'ssi1','ssi2','ssi3','ssi4','ssi5','ssi6','ssi7','ssi8', 
         'var1','var2','var3','var4','var5','var6','var7','var8',
         'wl1','wl2','wl3','wl4','wl5','wl6','wl7','wl8', 
         'movimiento']


#dataframe = pd.read_csv("totalFeatExtractedB1.csv", names=names)  # Usar con dataset de Berid
#dataframe = pd.read_csv("totalFeatExtracted-2.csv", names=names)   # Usar con data set propio
#dataframe = pd.read_csv("u15m1t4FeatExtracted.csv", names=names)   # Usar con data set propio
dataframe = pd.read_csv("totalFeatExtracted1.csv", names=names)   # Usar con data set propio

array = dataframe.values
#X = array[:,0:48]
#Y = array[:,48:49]
#test_size = 0.33
#seed = 7

#_, X_test, _, y_test = model_selection.train_test_split(X, Y, test_size=0.8)

X_test = array[:,0:48]
y_test = array[:,48:49]
#y_test = [[1],[1],[1],[1],[1]]

# load the model
mlp = load(open('modeloG3.pkl', 'rb'))
# load the scaler
scaler = load(open('escaladorG3.pkl', 'rb'))

# fit scaler on the training dataset
#scaler.fit(X_test)
# transform the training dataset
X_test_scaled = scaler.transform(X_test)


yhat = mlp.predict(X_test_scaled)
print("Y: ", y_test)
print("Y p: ", yhat)
# evaluate accuracy
acc = accuracy_score(y_test, yhat)
print('Test Accuracy:', acc)






