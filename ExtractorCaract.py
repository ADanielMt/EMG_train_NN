# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:38:23 2021

@author: AlejandroDaniel
"""

import time
import csv     #ADM
import numpy as np  #ADM
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt

contador=0
iduser=1
idtest=1
idmove=1

usercount=0;
testcount=0;
movcount=0;

clasificador =False #True#False
captura = False
Diferentes = True
mayor = [0,0,0,0,0,0,0,0]
menor = [1000,1000,1000,1000,1000,1000,1000,1000]
promedio = [0, 0, 0, 0, 0, 0, 0, 0]
moda = [[[0],[0]], [[0],[0]], [[0],[0]], [[0],[0]], [[0],[0]], [[0],[0]], [[0],[0]], [[0],[0]]]
maxi = [0,0,0,0,0,0,0,0]
veces = [0,0,0,0,0,0,0,0]
media= [0,0,0,0,0,0,0,0]
aux_mad = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
aux_cop = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
mavs_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
wl_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
aac_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
zc_data= [0,0,0,0,0,0,0,0]
kurt_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
skew_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
onset_index = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
onset_val = 0
sensor = 0
#no_datos = 1000
no_datos = 1000
long_dat = 0
#no_datos = 20 #2895
Ndatosclasif = 40
media_aux = [[0],[0],[0],[0],[0],[0],[0],[0]]
modaclases = [0]

results = []
features=[]
#media_auxT=np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
aux_Feat=np.zeros((1, 49))  #1,81
aux_mad2=np.zeros((no_datos-1, 8))
aux_mad3=np.zeros((no_datos-1, 8))

aux_cop2=np.zeros((no_datos, 8))
aux_cop3=np.zeros((no_datos, 8))



def featExtract(data_set, num_datos, label):
    
    #num_datos = 60
    wfl_data= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
############################# VALOR INTEGRADO ABSOLUTO Y VALOR MEDIO ABSOLUTO ######################################
                
    emg_rect=abs(data_set)
    print(emg_rect)
    
    iav_data=np.sum(emg_rect, axis=0)
    print("Valor Integrado Absoluto")
    print(iav_data)
    
    mav_data= np.mean(emg_rect, axis=0)
    print("Valor Medio Absoluto")
    print(mav_data)
                
######################### Raiz Media Cuadrada e Integral Simple Cuadrada  ############################################
        
    sq_data = emg_rect**2
    ssi_data = np.sum(sq_data, axis=0)
    print("Integral Simple Cuadrada")
    print(ssi_data)
    
    sumsq_data= np.mean(sq_data, axis=0)
    sqrt_data=np.sqrt(sumsq_data)
    print("Raiz Cuadrada Media")
    print(sqrt_data)
    
############################## Varianza ################################
                
    var_data=np.var(emg_rect, axis=0, ddof=1)
    print("Varianza")
    print(var_data)
                
############################# Longitud de Forma de Onda ###########################
    aux_add=0
    aux_subs=0
    
    for sd in range(8):
        for nd in range(num_datos-1):
            aux_subs = abs((data_set[nd+1][sd]) - (data_set[nd][sd]))
            aux_add=aux_add+aux_subs
        wfl_data[sd]=aux_add
        aux_add=0
        aux_subs=0 
    print("Longitud de Forma de Onda")
    print(wfl_data)
    
###################################################################################            
    features = np.concatenate([iav_data, mav_data, sqrt_data, ssi_data, var_data, wfl_data, label])             
    features = np.array([features])
    
    return(features)



for iduser in range(16):
    for idmove in range(4,9):       # in range(inicio, fin)
        for idtest in range(6):

            fileName="u"+str(iduser+1)+"m"+str(idmove+1)+"t"+str(idtest+1)+".csv"
            print(fileName)
            with open(fileName) as csvfile:   #Importar base de datos
                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # convertir a float
                for row in reader: # Cada fila como lista
                    results.append(row)   #Guardar en results
        
            dat_auxT = np.array(results)  # Convertir a arreglo de numpy
            
        ########################## Ventaneo y extracción de características ###################
            
            d_filtered=dat_auxT[:,:]
        
            movLabel=([idmove+1])
        
            feat_win_1 = featExtract(d_filtered[0:60,:], 60, movLabel)
            feat_win_2 = featExtract(d_filtered[60:120,:], 60, movLabel)
            feat_win_3 = featExtract(d_filtered[120:180,:], 60, movLabel)
           
        ################################## Matriz de características ######################
            feat_matrix = np.concatenate((feat_win_1, feat_win_2, feat_win_3), axis=0)
            print("Matriz de características")
            print(feat_matrix)
            
#            fileNameC="u"+str(iduser+1)+"m"+str(idmove+1)+"t"+str(idtest+1)+"featExtracted.csv"
#            with open(fileNameC,'w', newline='') as utm:
#                utm_writer=csv.writer(utm)
#                utm_writer.writerows(feat_matrix)
#            print("Matriz de características guardada")
#            print("Usuario ", iduser+1, " Movimiento ", idmove+1, " Prueba ", idtest+1)
            
            
            aux_Feat=np.concatenate((aux_Feat, feat_matrix), axis=0) # Para unir las carcaterísticas de cada prueba
            

            results=[0]
            results=[]
            media_auxT = np.array(0) 
            dat_auxT=np.array(0)
            d_filtered=np.array(0)
            feat_win_1 = np.array(0)
            feat_win_2 = np.array(0)
            feat_win_3 = np.array(0)
            feat_win_4 = np.array(0)

        
totalFeat = aux_Feat[1:,0:]           # Unir todas las características extraidas, excepto la primera fila,
                                      # ya que está llena de ceros
fileNameF="totalFeatExtracted4.csv"
with open(fileNameF,'w', newline='') as tfeat:
    tfeat_writer=csv.writer(tfeat)
    tfeat_writer.writerows(totalFeat)
    print("Matriz de características totales guardada")







