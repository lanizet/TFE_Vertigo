#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:14:01 2022

@author: anizetlaura & doyenastrid
"""


# Import of packages
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import csv
import pandas as pd
from scipy import stats

#Load the values of degree and betweenness centrality for all the participants
#---------------------------------------------------------------------------
degree = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/degree.csv", header=0)
degree = degree.drop("Unnamed: 0", axis=1)

between = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/between.csv", header=0)
between = between.drop("Unnamed: 0", axis=1)

columns = degree.columns
index = degree.index

nodes_left = columns[0:124]
nodes_right = columns[124:249]

#Statistical tests to see group differences based on degree and betweenness centrality
#---------------------------------------------------------------------------      
k = 0
for j in [nodes_left,nodes_right]:
    
    degree_tab = pd.DataFrame(columns = ["Statistical value", "p-value", "test type"])    
    between_tab = pd.DataFrame(columns = ["Statistical value", "p-value", "test type"]) 
    print(j)
    k += 1
    for i in j:   
        print(i)
        degree_shapiro_H = stats.shapiro(np.asarray(degree[0:7][i]))
        degree_shapiro_V_NC = stats.shapiro(np.asarray(degree[7:59][i]))
        degree_shapiro_V_C = stats.shapiro(np.asarray(degree[59:70][i]))

        between_shapiro_H = stats.shapiro(np.asarray(between[0:7][i]))
        between_shapiro_V_NC = stats.shapiro(np.asarray(between[7:59][i]))
        between_shapiro_V_C = stats.shapiro(np.asarray(between[59:70][i]))
        
        if degree_shapiro_H[1] > 0.05 and degree_shapiro_V_NC[1] > 0.05 and degree_shapiro_V_C[1] > 0.05:
            bartlett_result = stats.bartlett(degree[0:7][i], degree[7:59][i], degree[59:70][i])
            if bartlett_result[1] > 0.05:        
                anova_result = stats.f_oneway(np.asarray(degree[0:7][i]), np.asarray(degree[7:59][i]), np.asarray(degree[59:70][i]))
                new_elem = pd.DataFrame(data=np.array([[anova_result[0], anova_result[1], "ANOVA test"]]), columns=["Statistical value", "p-value", "test type"])
                degree_tab = pd.concat([degree_tab,new_elem], ignore_index=True)  
            else:
                kruskal_result = stats.kruskal(np.asarray(degree[0:7][i], dtype=np.float64), np.asarray(degree[7:59][i], dtype=np.float64), np.asarray(degree[59:70][i], dtype=np.float64))
                new_elem = pd.DataFrame(data=np.array([[kruskal_result[0], kruskal_result[1], "Kruskal-Wallis H-test"]]), columns=["Statistical value", "p-value", "test type"])
                degree_tab = pd.concat([degree_tab,new_elem], ignore_index=True)

        
        else:
            kruskal_result = stats.kruskal(np.asarray(degree[0:7][i], dtype=np.float64), np.asarray(degree[7:59][i], dtype=np.float64), np.asarray(degree[59:70][i], dtype=np.float64))
            new_elem = pd.DataFrame(data=np.array([[kruskal_result[0], kruskal_result[1], "Kruskal-Wallis H-test"]]), columns=["Statistical value", "p-value", "test type"])
            degree_tab = pd.concat([degree_tab,new_elem], ignore_index=True)


        if between_shapiro_H[1] > 0.05 and between_shapiro_V_NC[1] > 0.05 and between_shapiro_V_C[1] > 0.05:
            bartlett_result = stats.bartlett(between[0:7][i], between[7:59][i], between[59:70][i])
            if bartlett_result[1] > 0.05:        
                anova_result = stats.f_oneway(np.asarray(between[0:7][i]), np.asarray(between[7:59][i]), np.asarray(between[59:70][i]))
                new_elem = pd.DataFrame(data=np.array([[anova_result[0], anova_result[1], "ANOVA test"]]), columns=["Statistical value", "p-value", "test type"])
                between_tab = pd.concat([between_tab,new_elem], ignore_index=True)  
            else:
                kruskal_result = stats.kruskal(np.asarray(between[0:7][i], dtype=np.float64), np.asarray(between[7:59][i], dtype=np.float64), np.asarray(between[59:70][i], dtype=np.float64))
                new_elem = pd.DataFrame(data=np.array([[kruskal_result[0], kruskal_result[1], "Kruskal-Wallis H-test"]]), columns=["Statistical value", "p-value", "test type"])
                between_tab = pd.concat([between_tab,new_elem], ignore_index=True)
        else:
            kruskal_result = stats.kruskal(np.asarray(between[0:7][i], dtype=np.float64), np.asarray(between[7:59][i], dtype=np.float64), np.asarray(between[59:70][i], dtype=np.float64))
            new_elem = pd.DataFrame(data=np.array([[kruskal_result[0], kruskal_result[1], "Kruskal-Wallis H-test"]]), columns=["Statistical value", "p-value", "test type"])
            between_tab = pd.concat([between_tab,new_elem], ignore_index=True)
  
    nodes_l = np.arange(0,124,1)
    nodes_r = np.arange(124,249,1)

    if k == 1:
        degree_tab.insert(0, "Nodes", pd.DataFrame(nodes_l))
        between_tab.insert(0, "Nodes", pd.DataFrame(nodes_l))
        degree_tab = degree_tab.sort_values(by=["p-value"])
        between_tab = between_tab.sort_values(by=["p-value"])
        degree_tab.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_left.csv")
        between_tab.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_left.csv")
    if k == 2:
        degree_tab.insert(0, "Nodes", pd.DataFrame(nodes_r))
        between_tab.insert(0, "Nodes", pd.DataFrame(nodes_r))
        degree_tab = degree_tab.sort_values(by=["p-value"])
        between_tab = between_tab.sort_values(by=["p-value"])
        degree_tab.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_right.csv")
        between_tab.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_right.csv")