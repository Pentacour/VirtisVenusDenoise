# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:23:07 2022

@author: Ramon
"""
import os
import sys
import pandas as pd
import numpy as np


#%% 0100_1000
SAVED_MODEL_UNET = "0100_1000-64-unet-xxn3"
SAVED_MODEL_RESNET = "0100_1000-64-resnet-xxh-2"
SAVED_MODEL_CONVSIM = "0100_1000-64-convsim-xxc3-2"
SAVED_MODEL_AECONNECT = "0100_1000-64-aeconnect-xxe4"
SAVED_MODEL_RESULTS = "0100_1000-64-"

#%% 0001_0010
SAVED_MODEL_UNET = "0001_0010-64-unet-xxn3"
SAVED_MODEL_RESNET = "0001_0010-64-resnet-xxh"
SAVED_MODEL_CONVSIM = "0001_0010-64-convsim-xxc3"
SAVED_MODEL_AECONNECT = "0001_0010-64-aeconnect-xe4"
SAVED_MODEL_RESULTS = "0001_0100-64-"

#%% 0010_1000
SAVED_MODEL_UNET = "0010_1000-64-unet-xxn4"
SAVED_MODEL_RESNET = "0010_1000-64-resnet-xxh"
SAVED_MODEL_CONVSIM = "0010_1000-64-convsim-xxc3"
SAVED_MODEL_AECONNECT = "0010_1000-64-aeconnect-xxe4"
SAVED_MODEL_RESULTS = "0010_1000-64-"

#%% 0001_0100
SAVED_MODEL_UNET = "0001_0100-64-unet-xxn4"
SAVED_MODEL_RESNET = "0001_0100-64-resnet-xxh"
SAVED_MODEL_CONVSIM = "0001_0100-64-convsim-xc4" #! 
SAVED_MODEL_AECONNECT = "0001_0100-64-aeconnect-xxe4"
SAVED_MODEL_RESULTS = "0001_0100-64-"


#%% BEST
DEST_TESTS_UNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_UNET))
DEST_TESTS_RESNET = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESNET))
DEST_TESTS_CONVSIM = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_CONVSIM))
DEST_TESTS_AECONNECT = os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_AECONNECT))

df_unet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_UNET, 'metrics_' +  SAVED_MODEL_UNET + '.csv')), sep=';')
df_resnet = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_RESNET, 'metrics_' +  SAVED_MODEL_RESNET + '.csv')), sep=';')
df_convsim = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_CONVSIM, 'metrics_' +  SAVED_MODEL_CONVSIM + '.csv')), sep=';')
df_aeconnect = pd.read_csv(os.path.abspath(os.path.join(DEST_TESTS_AECONNECT, 'metrics_' +  SAVED_MODEL_AECONNECT + '.csv')), sep=';')

#%% BEST

print("BEST " + SAVED_MODEL_RESULTS)

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
    
    with open(os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESULTS + str(metric_index) + '_BEST.csv')), 'w', newline='') as file_csv:

        file_csv.write("image;best\n")
        
        for i in range( len(df_unet)):
            file_csv.write( df_unet.iloc[i][0] + ";")
            file_csv.write( str(images_best[i]) + "\n")
 
#%% WORST

print("WORST " + SAVED_MODEL_RESULTS)

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
            selected_value = max(values)
        else:
            selected_value = min(values)

        selected_index = values.index(selected_value)
        metric_scores.append(selected_index)
        
        images_best[index] = selected_index

    print("SCORE:"+str(metric_index))        
    for i in range(4):
        print(str(i) + "->" + str(metric_scores.count(i)) + " / " + str(len(metric_scores)))
    
    with open(os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESULTS + str(metric_index) + '_WORST.csv')), 'w', newline='') as file_csv:

        file_csv.write("image;worst\n")
        
        for i in range( len(df_unet)):
            file_csv.write( df_unet.iloc[i][0] + ";")
            file_csv.write( str(images_best[i]) + "\n")

#%% SECOND_BEST

# print("SECOND")

# for metric_index in range(7,10):
#     metric_scores = []
#     images_best = np.empty(len(df_unet))
#     with_no_values = 0
    
#     for index, row in df_unet.iterrows():
#         values = [0,0,0,0]
        
#         if df_unet.iloc[index][0] != df_resnet.iloc[index][0] \
#             or df_unet.iloc[index][0] !=  df_convsim.iloc[index][0] \
#             or df_unet.iloc[index][0] !=  df_aeconnect.iloc[index][0]:
            
#             print("Error: Differnet order in images")
#             sys.exit(-1)
        
#         values[0] = df_unet.iloc[index][metric_index]
#         values[1] = df_resnet.iloc[index][metric_index]
#         values[2] = df_convsim.iloc[index][metric_index]
#         values[3] = df_aeconnect.iloc[index][metric_index]
        
#         values[0] = float(values[0].replace(",","."))
#         values[1] = float(values[1].replace(",","."))
#         values[2] = float(values[2].replace(",","."))
#         values[3] = float(values[3].replace(",","."))
        
#         if metric_index == 7:
#             ordered = set(values)
#             ordered.remove(min(ordered))
#             selected_value = min(ordered)
#         else:
#             selected_value = max(values)
            
#             ordered = set(values)
#             ordered.remove(max(ordered))
#             if len(ordered) > 0:
#                 selected_value = max(ordered)
#             else:
#                 with_no_values +=1


#         selected_index = values.index(selected_value)
#         metric_scores.append(selected_index)
        
#         images_best[index] = selected_index

#     print("SCORE:"+str(metric_index))       
#     if with_no_values > 0:
#         print("With no values:" + str(with_no_values))
#     for i in range(4):
#         print(str(i) + "->" + str(metric_scores.count(i)) + " / " + str(len(metric_scores)))
    
#     with open(os.path.abspath(os.path.join('../../../out_tests/', SAVED_MODEL_RESULTS + str(metric_index) + '_SECOND.csv')), 'w', newline='') as file_csv:

#         file_csv.write("image;best\n")
        
#         for i in range( len(df_unet)):
#             file_csv.write( df_unet.iloc[i][0] + ";")
#             file_csv.write( str(images_best[i]) + "\n")
 
