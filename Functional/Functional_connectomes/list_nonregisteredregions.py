# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:56:05 2022

@author: anizetlaura & doyenastrid
"""
import numpy as np
import json
import matplotlib.pyplot as plt

path='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_connectome/'

with open(path+"registration_functionnel.json", "r") as file:
    data = json.load(file)

    
patient_list = ["H_"+str(_) for _ in range(7)] + ["V_"+str(_) for _ in range(0,12)] + ["V_"+str(_) for _ in range(13,54)] + ["C_"+str(_) for _ in range(0,4)] + ["C_"+str(_) for _ in range(5,13)]

nonregisteredregions = []
for i in patient_list:
    if data[i] != '[]':
        for j in range (0, len(data[i])):
            if data[i][j] not in nonregisteredregions:
                nonregisteredregions.append(data[i][j])
                
nonregisteredregions.sort(reverse=True)

with open(path+"sorted_nonregisteredlabels_functionnel.json", "a") as file:
    json.dump(nonregisteredregions, file)

for i in patient_list:
    matrix = np.loadtxt(path+'%s/func_connectome_%s.txt'%(i,i))

    print(matrix.shape)
    labels_to_remove = np.asarray(nonregisteredregions)-1
    print(labels_to_remove)
    conn_matrix = np.delete(matrix, labels_to_remove, 0)
    conn_matrix = np.delete(conn_matrix, labels_to_remove, 1)

    print(conn_matrix.shape)

    plt.matshow(conn_matrix)
    plt.savefig(path+"%s/func_connectome_cleaned_%s.png"%(i,i))
    np.savetxt(path+"%s/func_connectome_cleaned_%s.txt"%(i,i), conn_matrix, fmt='%f')


