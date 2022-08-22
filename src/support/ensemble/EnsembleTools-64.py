# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:23:07 2022

@author: Ramon
"""
import os
import sys
import pandas as pd
import numpy as np


SAVED_MODEL_UNET = "0100_1000-64-unet-xn3"
SAVED_MODEL_RESNET = "0100_1000-64-unet-xn3"
SAVED_MODEL_CONVSIM = "0100_1000-64-convsim-xc3"
SAVED_MODEL_AECONNECT = "0100_1000-64-aeconnect-xe4"
SAVED_MODEL_ENSEMBLE = "0100_1000-64-aeconnect-xe4"

DEST_TESTS_UNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_UNET))
DEST_TESTS_RESNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESNET))
DEST_TESTS_CONVSIM = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_CONVSIM))
DEST_TESTS_AECONNECT = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_AECONNECT))
DEST_TESTS_ENSEMBLE = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_ENSEMBLE))

df_unet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_UNET, 'metrics_' +  SAVED_MODEL_UNET + '.csv')), sep=';')
df_resnet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_RESNET, 'metrics_' +  SAVED_MODEL_RESNET + '.csv')), sep=';')
df_convsim = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_CONVSIM, 'metrics_' +  SAVED_MODEL_CONVSIM + '.csv')), sep=';')
df_aeconnect = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_AECONNECT, 'metrics_' +  SAVED_MODEL_AECONNECT + '.csv')), sep=';')
df_ensemble = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_ENSEMBLE, 'metrics_' +  SAVED_MODEL_ENSEMBLE + '.csv')), sep=';')

#%%

images_best = np.empty(len(df_unet))

for score_index in range(7,10):
    scores = []
    
    for index, row in df_unet.iterrows():
        values = [0,0,0,0]
        
        if df_unet.iloc[index][0] != df_resnet.iloc[index][0] \
            or df_unet.iloc[index][0] !=  df_convsim.iloc[index][0] \
            or df_unet.iloc[index][0] !=  df_aeconnect.iloc[index][0]:
            
            print("Error: Differnet order in images")
            sys.exit(-1)
        
        values[0] = df_unet.iloc[index][score_index]
        values[1] = df_resnet.iloc[index][score_index]
        values[2] = df_convsim.iloc[index][score_index]
        values[3] = df_aeconnect.iloc[index][score_index]
        #values[4] = df_ensemble.iloc[index][score_index]
        
        values[0] = float(values[0].replace(",","."))
        values[1] = float(values[1].replace(",","."))
        values[2] = float(values[2].replace(",","."))
        values[3] = float(values[3].replace(",","."))
        
        if index == 7:
            selected_value = min(values)
        else:
            selected_value = max(values)

        selected_index = values.index(selected_value)
        scores.append(selected_index)
        
        images_best[index] = selected_index

    print("SCORE:"+str(score_index))        
    for i in range(4):
        print(str(i) + "->" + str(scores.count(i)) + " / " + str(len(scores)))
    
    
print(images_best)
