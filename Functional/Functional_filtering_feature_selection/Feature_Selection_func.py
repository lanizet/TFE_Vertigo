#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:07:05 2022


@author: anizetlaura & doyenastrid
"""

#---------------------------------------------------------------------------
# Import of packages
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import json
import csv
import sklearn
import sklearn.feature_selection
from scipy import stats

path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"
features_table = pd.read_csv(path+"functional_features/Features_table_nonnul_functionnel.csv")
features_table = features_table.drop("Unnamed: 0", axis=1)

features_table = features_table.abs()
print(len(features_table.index))

target = np.zeros(72)
target[7:60] = 1
target[60:72] = 2

"""
#---------------------------------------------------------------------------
PEARSON CORRELATION
#---------------------------------------------------------------------------
"""
def select_features_correlation(X, Y, thres):
    ## Return the correlation matrix, the name of the regions that are correlated to the target with a correlation value above the threshold and the correlation values by ascending order
    
    cor = np.corrcoef(X.T, Y.T)
    names = X.columns.tolist()
    comp = np.abs(cor[-1])[:-1]
    index = np.argwhere(comp>thres)
    index = np.ravel(index)
    regions_relevant=[]
    for i in range (0,len(index)):
        regions_relevant.append(names[index[i]])
    regions_relevant = np.asarray(regions_relevant)

    X_filtered = X[regions_relevant]

    return X_filtered, regions_relevant, comp

threshold = [0.3]
inc = ["Three"]
for i in range(len(threshold)):
    features_corr, regions_corr, corr_values = select_features_correlation(features_table, target, threshold[i])

    print(len(regions_corr))

    pd.DataFrame(regions_corr).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_corr_func_%s.p"%inc[i])#.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_corr_%s.csv"%inc[i])#%threshold[i], index=False)
    pd.DataFrame(features_corr).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/features_selected_corr_func_%s.p"%inc[i])#%threshold[i], index=False)
    pd.DataFrame(corr_values).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/values_corr_func_%s.p"%inc[i])#%threshold[i], index=False)


"""
#---------------------------------------------------------------------------
MUTUAL INFORMATION
#---------------------------------------------------------------------------
"""
def mutual_info(X, y, percentage, iteration):
  ## Return the mutual information matrix, the name of the regions that have a MI with the target above the threshold and the MI values by ascending order
    
  tab_MI = []
  names = X.columns.tolist()
  lgr = len(names)
  n_features = int((len(names)*percentage))
  tab = np.zeros((iteration,lgr))
  for j in range(iteration):
      print(j)
      MI = sklearn.feature_selection.mutual_info_classif(np.asarray(X), np.asarray(y))
      tab_MI.append(MI)
      MI_sorted = np.argsort(MI)[::-1] #Sort the MI indices in descending order: first the index of the largest MI 
      print(MI_sorted)
      for i in range(lgr):
         tab[j, MI_sorted[i]] = i #The smallest MI is given a higher value while keeping the order in columns
  index = np.argsort(np.sum(tab, axis = 0))[::-1] #Sorts in descending order the indices of the largest sums (= feature with the smallest MI over the 1000 iterations) 
  i = lgr
  j = 0
  while i > n_features:
    X = X.drop(names[index[j]],1) #Remove the features that had the smallest MI
    i = i-1
    j = j+1
    
  idx_inv = np.asarray(index[::-1], dtype=int)[0:n_features]
  name_sorted = []
  tab_sum = np.sum(np.asarray(tab_MI), axis=0)
  print(tab_sum)
  MI_values = []
  for i in tqdm(idx_inv):
      name_sorted.append(names[i])
      MI_values.append(tab_sum[i])
      
  X_filtered = X[name_sorted]
  MI_values = np.asarray(MI_values)/iteration
      
  return X_filtered, name_sorted, MI_values

inc = ["Three"]

for i in range(len(inc)):
    features_redundant = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/svc/features_kept_redundant_%s.csv"%(inc[i],inc[i]))
    features_redundant = features_redundant.drop("Unnamed: 0", axis=1)
    print(features_redundant.shape[1]-1)
    print(features_table.shape[1])
    perc = (features_redundant.shape[1]-1)/features_table.shape[1]
    print(perc)

    features_MI, regions_MI, MI_values = mutual_info(features_table, target, perc, 50)

    print(len(regions_MI))

    pd.DataFrame(regions_MI).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_MI_func_%s.p"%inc[i])#.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_corr_%s.csv"%inc[i])#%threshold[i], index=False)
    pd.DataFrame(features_MI).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/features_selected_MI_func_%s.p"%inc[i])#%threshold[i], index=False)
    pd.DataFrame(MI_values).to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/values_MI_func_%s.p"%inc[i])#%threshold[i], index=False)