# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:23:07 2022

@author: Ramon
"""
import os
import sys
import pandas as pd
import numpy as np


SAVED_MODEL_UNET = "0100_1000-64-unet-xn3_train"
SAVED_MODEL_RESNET = "0100_1000-64-resnet-h_train"
SAVED_MODEL_CONVSIM = "0100_1000-64-convsim-xc3_train"
SAVED_MODEL_AECONNECT = "0100_1000-64-aeconnect-xe4_train"
SAVED_MODEL_ENSEMBLE = "0100_1000-64-aeconnect-xe4_train"

SAVED_MODEL_RESULTS = "0100_1000-64-train"

DEST_TESTS_UNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_UNET))
DEST_TESTS_RESNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESNET))
DEST_TESTS_CONVSIM = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_CONVSIM))
DEST_TESTS_AECONNECT = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_AECONNECT))
DEST_TESTS_ENSEMBLE = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_ENSEMBLE))

df_unet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_UNET, 'metrics_' +  \
                                                     SAVED_MODEL_UNET.replace('_train','') + '.csv')), sep=';')
df_resnet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_RESNET, 'metrics_' + \
                                                     SAVED_MODEL_RESNET.replace('_train','') + '.csv')), sep=';')
df_convsim = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_CONVSIM, 'metrics_' + \
                                                     SAVED_MODEL_CONVSIM.replace('_train','') + '.csv')), sep=';')
df_aeconnect = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_AECONNECT, 'metrics_' +  \
                                                     SAVED_MODEL_AECONNECT.replace('_train','') + '.csv')), sep=';')
df_ensemble = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_ENSEMBLE, 'metrics_' + \
                                                     SAVED_MODEL_ENSEMBLE.replace('_train','') + '.csv')), sep=';')

#%%

for metric_index in range(7,10):
    metric_scores = []
    images_best = np.empty(len(df_unet))
    
    for index, row in df_unet.iterrows():
        values = [0,0,0,0]
        
        if df_unet.iloc[index][0] != df_resnet.iloc[index][0] \
            or df_unet.iloc[index][0] !=  df_convsim.iloc[index][0] \
            or df_unet.iloc[index][0] !=  df_aeconnect.iloc[index][0]:
            
            print("Error: Differnet order in images")
            sys.exit(-1)
        
        values[0] = df_unet.iloc[index][metric_index]
        values[1] = df_resnet.iloc[index][metric_index]
        values[2] = df_convsim.iloc[index][metric_index]
        values[3] = df_aeconnect.iloc[index][metric_index]
        
        values[0] = float(values[0].replace(",","."))
        values[1] = float(values[1].replace(",","."))
        values[2] = float(values[2].replace(",","."))
        values[3] = float(values[3].replace(",","."))
        
        if metric_index == 7:
            selected_value = min(values)
        else:
            selected_value = max(values)

        selected_index = values.index(selected_value)
        metric_scores.append(selected_index)
        
        images_best[index] = selected_index

    print("SCORE:"+str(metric_index))        
    for i in range(4):
        print(str(i) + "->" + str(metric_scores.count(i)) + " / " + str(len(metric_scores)))
    
    with open(os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESULTS + str(metric_index) + '.csv')), 'w', newline='') as file_csv:

        file_csv.write("image;best\n")
        
        for i in range( len(df_unet)):
            file_csv.write( df_unet.iloc[i][0] + ";")
            file_csv.write( str(images_best[i]) + "\n")
 
