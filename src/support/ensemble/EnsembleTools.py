# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:23:07 2022

@author: Ramon
"""
import os
import pandas as pd


SAVED_MODEL_UNET = "0100_1000-256-unet-b-nadam-b"
SAVED_MODEL_RESNET = "0100_1000-256-resnet-b"
SAVED_MODEL_CONVSIM = "0100_1000-256-convsim-b"
SAVED_MODEL_AECONNECT = "0100_1000-256-aeconnect-a"
SAVED_MODEL_ENSEMBLE = "0100_1000-256-ensemble-a"

DEST_TESTS_UNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_UNET))
DEST_TESTS_RESNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESNET))
DEST_TESTS_CONVSIM = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_CONVSIM))
DEST_TESTS_AECONNECT = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_AECONNECT))
DEST_TESTS_ENSEMBLE = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_ENSEMBLE))

df_unet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_UNET, 'scores_' +  SAVED_MODEL_UNET + '.csv')), sep=';')
df_resnet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_RESNET, 'scores_' +  SAVED_MODEL_RESNET + '.csv')), sep=';')
df_convsim = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_CONVSIM, 'scores_' +  SAVED_MODEL_CONVSIM + '.csv')), sep=';')
df_aeconnect = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_AECONNECT, 'scores_' +  SAVED_MODEL_AECONNECT + '.csv')), sep=';')
df_ensemble = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_ENSEMBLE, 'scores_' +  SAVED_MODEL_ENSEMBLE + '.csv')), sep=';')

#%%
values = [0,0,0,0,0]

for score_index in range(1,4):
    scores = []
    
    for index, row in df_unet.iterrows():
        values[0] = row[score_index]
    
        values[1] = df_resnet.iloc[index][score_index]
        values[2] = df_convsim.iloc[index][score_index]
        values[3] = df_aeconnect.iloc[index][score_index]
        values[4] = df_ensemble.iloc[index][score_index]
        
        max_value = max(values)
        max_index = values.index(max_value)
        
        scores.append(max_index)

    print("SCORE:"+str(score_index))        
    for i in range(5):
        print(str(i) + "->" + str(scores.count(i)) + " / " + str(len(scores)))
    
    
