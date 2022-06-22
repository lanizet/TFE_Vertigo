# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:37:29 2022

@author: anizetlaura & doyenastrid
"""

import pandas as pd
import numpy as np

#Creation for each hemisphere of a table in with the p-value are in increasing order and in which the Benjamini threshold is computed
#-----------------------------------------------------------------------
degree_left = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_left.csv")
degree_left = degree_left.drop("Unnamed: 0", axis=1)
degree_left = degree_left.sort_values(by = ['p-value'], axis=0)

between_left = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_left.csv")
between_left = between_left.drop("Unnamed: 0", axis=1)
between_left = between_left.sort_values(by = ['p-value'], axis=0)

degree_right = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_right.csv")
degree_right = degree_right.drop("Unnamed: 0", axis=1)
degree_right = degree_right.sort_values(by = ['p-value'], axis=0)

between_right = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_right.csv")
between_right = between_right.drop("Unnamed: 0", axis=1)
between_right = between_right.sort_values(by = ['p-value'], axis=0)

benjamini_values_left = np.zeros_like(degree_left['p-value'])
benjamini_values_right = np.zeros_like(degree_right['p-value'])

for i in range(len(degree_left)):
    benjamini_values_left[i] = (0.1*(i+1))/len(degree_left['p-value'])
for i in range(len(degree_right)):
    benjamini_values_right[i] = (0.1*(i+1))/len(degree_right['p-value'])
    
degree_left.insert(degree_left.shape[1]-1,"Benjamini-values",pd.DataFrame(benjamini_values_left))
between_left.insert(between_left.shape[1]-1,"Benjamini-values",pd.DataFrame(benjamini_values_left))
degree_right.insert(degree_right.shape[1]-1,"Benjamini-values",pd.DataFrame(benjamini_values_right))
between_right.insert(between_right.shape[1]-1,"Benjamini-values",pd.DataFrame(benjamini_values_right))

#Save each table of p-values in a csv file
#-----------------------------------------------------------------------
degree_left.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_left_benjamini.csv")
between_left.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_left_benjamini.csv")
degree_right.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_degree_right_benjamini.csv")
between_right.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_network_measures/stats_between_right_benjamini.csv")
