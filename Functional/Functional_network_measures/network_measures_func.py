#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:14:01 2022

@author: anizetlaura & doyenastrid
"""

#---------------------------------------------------------------------------
# Import of packages
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import csv
import pandas as pd
from scipy import stats
import networkx as nx

#---------------------------------------------------------------------------
# Import of packages
#---------------------------------------------------------------------------

path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"

#Divisions of the patients in the three groups (Control, vertigo and compensated)
#----------------------------------------------------------------------------
patients_sains = ["H_"+str(_) for _ in range(7)]
patients_vertiges = ["V_"+str(_) for _ in range(12)] + ["V_"+str(_) for _ in range(13,54)] 
patients_compenses = ["C_"+str(_) for _ in range(0,4)] + ["C_"+str(_) for _ in range(5,13)]

#Computation of graph theory metrics for each group
#----------------------------------------------------------------------------
degree_sains = []
between_sains = []
path_length_sains = []
clustering_sains = []
for i in range(len(patients_sains)):
    matrix =  np.loadtxt(path+"functional_connectome/%s/func_connectome_cleaned_%s.txt"%(patients_sains[i],patients_sains[i]))
    ztransform = np.arctanh(matrix)
    ztransform[ztransform<0.25] = 0
    graph = nx.from_numpy_matrix(ztransform)
    degree_sains.append(nx.degree_centrality(graph))
    between_sains.append(nx.betweenness_centrality(graph))


degree_vertiges = []
between_vertiges = []
path_length_vertiges = []
clustering_vertiges = []
for i in range(len(patients_vertiges)):
    matrix =  np.loadtxt(path+"functional_connectome/%s/func_connectome_cleaned_%s.txt"%(patients_vertiges[i],patients_vertiges[i]))
    ztransform = np.arctanh(matrix)
    ztransform[ztransform<0.25] = 0
    graph = nx.from_numpy_matrix(ztransform)
    degree_vertiges.append(nx.degree_centrality(graph))
    between_vertiges.append(nx.betweenness_centrality(graph))
    
degree_compenses = []
between_compenses = []
path_length_compenses = []
clustering_compenses = []
for i in range(len(patients_compenses)):
    matrix =  np.loadtxt(path+"functional_connectome/%s/func_connectome_cleaned_%s.txt"%(patients_compenses[i],patients_compenses[i]))
    ztransform = np.arctanh(matrix)
    ztransform[ztransform<0.25] = 0
    graph = nx.from_numpy_matrix(ztransform)
    degree_compenses.append(nx.degree_centrality(graph))
    between_compenses.append(nx.betweenness_centrality(graph))

#Creation of two csv files containing the values of degree betweennees and centrality for all the participants
#----------------------------------------------------------------------------
degree = pd.concat([pd.DataFrame(degree_sains),pd.DataFrame(degree_vertiges),pd.DataFrame(degree_compenses)],ignore_index=True)
degree.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/connectivity_measures/func/degree.csv")

between = pd.concat([pd.DataFrame(between_sains),pd.DataFrame(between_vertiges),pd.DataFrame(between_compenses)],ignore_index=True)
between.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/connectivity_measures/func/between.csv")


