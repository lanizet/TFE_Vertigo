#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:07:05 2022


@author: anizetlaura & doyenastrid
"""


# Import of packages
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import json
import csv

#---------------------------------------------------------------------------
# Code to run
#---------------------------------------------------------------------------
if __name__=='__main__':

    path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"
    patients = ["H_"+str(_) for _ in range(7)] + ["V_"+str(_) for _ in range(12)] + ["V_"+str(_) for _ in range(13,54)] + ["C_"+str(_) for _ in range(0,4)] + ["C_"+str(_) for _ in range(5,13)] 
    
    with open(path+"functional_connectome/sorted_nonregisteredlabels_functionnel.json") as file:
        labels_nonregistered = json.load(file)

    regions_names = pd.read_csv(path+"regions_name.csv")
    regions_names = np.asarray(regions_names)
    regions_names = np.delete(regions_names, labels_nonregistered, 0)
    regions_names = np.delete(regions_names, 0, 0)
    columns_name = ["Patient_name"]
    #Labels of the feature tables
    for i in range(len(regions_names)):
        for j in range(i+1,len(regions_names)):
            new_name = regions_names[i] + "To" + regions_names[j]
            columns_name.append(new_name[0])

    table_features = pd.DataFrame(columns = columns_name)
    
    #Give the values to the appropiate patient and the appropiate label
    for k in range(len(patients)):
        matrice = np.loadtxt(path+"functional_connectome/%s/func_connectome_cleaned_%s.txt"%(patients[k],patients[k]))
        values = [patients[k]]
        for l in range(len(regions_names)):
            for m in range(l+1,len(regions_names)):
                values.append(matrice[l][m])
        new_row = pd.DataFrame(data=[np.asarray(values)], columns=columns_name)
        table_features = pd.concat([table_features,new_row], ignore_index=True)
    
    table_features.to_csv(path+"functional_features/Features_table_functionnel.csv")
    table_features.to_pickle(path+"functional_features/Features_table_functionnel.p")

    features_table = pd.read_pickle(path+"functional_features/Features_table_functionnel.p")
    
    features_table = features_table.drop("Patient_name", axis=1)
    
    for i in features_table.columns:
        countH = len(features_table[i].iloc[0:7][features_table[i].iloc[0:7] == 0])
        countV = len(features_table[i].iloc[7:60][features_table[i].iloc[7:60] == 0])
        countVC = len(features_table[i].iloc[60:72][features_table[i].iloc[60:72] == 0])
        print(countH,countV,countVC)
        if countH > 3 and countV > 31 and countVC > 6 :
            print("Removed %s"%i)
            features_table = features_table.drop(i, axis = 1)

    features_table.to_csv(path+"functional_features/Features_table_nonnul_functionnel.csv")
    features_table.to_pickle(path+"functional_features/Features_table_nonnul_functionnel.p")
