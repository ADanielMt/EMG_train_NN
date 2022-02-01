# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 03:04:50 2019

@author: AlejandroDaniel
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier   
from sklearn.metrics import accuracy_score

# Location of dataset
#featuresdata = pd.read_csv("totalFeat.csv")
#featuresdata = pd.read_csv("totalFeatReduced.csv")

# Assign colum names to the dataset
names = ['iav1','iav2','iav3','iav4','iav5','iav6','iav7','iav8', 
         'mav1','mav2','mav3','mav4','mav5','mav6','mav7','mav8',
         'sqrt1','sqrt2','sqrt3','sqrt4','sqrt5','sqrt6','sqrt7','sqrt8', 
         'ssi1','ssi2','ssi3','ssi4','ssi5','ssi6','ssi7','ssi8', 
         'var1','var2','var3','var4','var5','var6','var7','var8',
         'wl1','wl2','wl3','wl4','wl5','wl6','wl7','wl8', 
         'movimiento']

# Read dataset to pandas dataframe
feature_data = pd.read_csv("totalFeatExtracted2.csv", names=names)  

array = feature_data.values
X = array[:,0:48]
Y = array[:,48:49]

# Assign data from first four columns to X variable
x = feature_data.iloc[:, 0:48]

# Assign data from first fifth columns to y variable
y = feature_data.iloc[:, 48:49]

 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)  

scaler = StandardScaler()  
scaler.fit(X_train)
scaler.fit(X_test)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
      
mlp = MLPClassifier(hidden_layer_sizes=(24, 20),activation='logistic', solver='adam', max_iter=1750)#(100, 80, 40)
# 1150 o 1100 funciona bien  -- 1300 (24, 18, 24)
#mlp.fit(X_train, y_train.values.ravel())                                
mlp.fit(X_train, y_train.ravel())                                                      
                
print("Entrenamiento K10")
pickle.dump(mlp, open('modeloK10.pkl', 'wb'))
# save the scaler
pickle.dump(scaler, open('escaladorK10.pkl', 'wb'))
# K8 es el mejor hasta ahora (24, 29)...(1450)
# K9 es mejor que K8... (24, 20)...(1750)
                                                               
                                                                
predictions = mlp.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  

acc = accuracy_score(y_test, predictions)
print('Test Accuracy:', acc)

