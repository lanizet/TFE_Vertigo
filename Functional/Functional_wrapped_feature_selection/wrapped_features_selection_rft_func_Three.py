#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:07:05 2022


@author: anizetlaura & doyenastrid
"""

#---------------------------------------------------------------------------
# Import of packages
#---------------------------------------------------------------------------

from ensurepip import bootstrap
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import json
import csv
import sklearn
import sklearn.feature_selection
from scipy import stats
import sklearn.model_selection
import sklearn.pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sklearn.ensemble
from joblib import Parallel, delayed


#---------------------------------------------------------------------------
# Function definition
#---------------------------------------------------------------------------


def machine_learning_func(features, target, num_feats_keep, model_chosen, params):
    # Use a wrapper method to select features with forward direction
    # INPUT : - features : dataframe containing the features 
    #         - target : dataframe containing the target
    #         - num_feats_keep : array containing different percentages of features kept
    #         - model_chosen : the model used to select the features
    #         - params : the best parameters for the model
    # OUTPUT : - table_index : list of arrays containing the indices of the features selected for the different percentages of features kept
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    table_dtf = pd.DataFrame(columns=["num_features_kept", "indices"])
    for j in num_feats_keep:
        if model_chosen == "svc":
            model = sklearn.svm.LinearSVC(C = params)
        else:
            model = sklearn.ensemble.RandomForestClassifier(n_estimators = params, bootstrap=False)             
        SFS = sklearn.feature_selection.SequentialFeatureSelector(estimator=model, n_features_to_select=j, direction='forward', scoring='balanced_accuracy', cv=4, n_jobs=6)
        pipe = sklearn.pipeline.Pipeline([("Scaler", scaler), ("Method", SFS)])
        y_score = pipe.fit(features, target)
        selector = pipe.named_steps['Method']
        index_Selected = SFS.get_support(indices=True)
        new_line = pd.DataFrame(data=[[j, index_Selected]], columns=["num_features_kept", "indices"])
        table_dtf = pd.concat([table_dtf, new_line], ignore_index=True)
                         
    return table_dtf

def find_best_params_rft(X_train, y_train, model_, hyper_params_grid):
    # Determine the best parameters to use depending on the model
    # INPUT : - X_train : dataframe containing the features to train
    #         - y_train : dataframe containing the target to train 
    #         - model_ : the model used
    #         - hyper_params_grid : dictionnary of the parameters to tests in the GridSearch
    # OUTPUT : - grid.best_params_ : dictionnary of the parameters giving the best performance 

    grid = sklearn.model_selection.GridSearchCV(estimator = model_, param_grid=hyper_params_grid, cv=5, scoring='balanced_accuracy', n_jobs=6)
    grid.fit(X_train, y_train) 
    return grid.best_params_

def comparaison_accuracy(data,target,model_chosen, params):
    # Return the different accuracies of the model with the weights associated to each features
    # INPUT : - data : dataframe containing the features 
    #         - target : dataframe containing the target  
    #         - model_chosen : the model used
    #         - params : parameter returned from the find_best_params function 
    # OUTPUT : - accuracy_blcd_train_mean : mean value of the accuracies of the model on the train set with the balanced score
    #          - accuracy_blcd_test_mean : mean value of the accuracies of the model on the test set with the balanced score
    #          - accuracy_train_mean : mean value of the accuracies of the model on the train set with the non-balanced score
    #          - accuracy_test_mean : mean value of the accuracies of the model on the test set with the non-balanced score
    #          - weights : list of arrays containing the weights associated to each features when fitting the model
    #          - data.columns : array containing the names of the features     

    kfold = sklearn.model_selection.StratifiedKFold(5)
    scaler = sklearn.preprocessing.MinMaxScaler()
    accuracy_blcd_train = []
    accuracy_blcd_test = []
    accuracy_train = []
    accuracy_test = []
    weights = []
    for trn_idx, tst_idx in kfold.split(data, target):
        #Training Set
        X_train = data.iloc[trn_idx,:]
        Y_train = np.ravel(np.asarray(pd.DataFrame(target).iloc[trn_idx]))
        #Test Set    
        X_test = data.iloc[tst_idx,:]
        Y_test = np.ravel(np.asarray(pd.DataFrame(target).iloc[tst_idx]))
        if model_chosen == "svc":
            model = sklearn.svm.LinearSVC(params)
        else:
            model = sklearn.ensemble.RandomForestClassifier(n_estimators = params, bootstrap=False)
        pipe = sklearn.pipeline.Pipeline([("Scaler", scaler), ("Method", model)])
        y_score = pipe.fit(X_train, Y_train)
        selector = pipe.named_steps['Method']
        #Weights
        weights.append(np.abs(selector.feature_importances_))
        #Accuracy of the model on the train set with the balanced score
        Y_pred_train_blcd = pipe.predict(X_train) 
        accuracy_blcd_train.append(balanced_accuracy_score(Y_train, Y_pred_train_blcd))
        #Accuracy of the model on the test set with the balanced score
        Y_true, Y_pred_blcd = Y_test, pipe.predict(X_test)   
        accuracy_blcd_test.append(balanced_accuracy_score(Y_true, Y_pred_blcd))
        #Accuracy of the model on the train set with the non-balanced score
        Y_pred_train = pipe.predict(X_train) 
        accuracy_train.append(accuracy_score(Y_train, Y_pred_train))
        #Accuracy of the model on the test set with the non-balanced score
        Y_true, Y_pred = Y_test, pipe.predict(X_test)   
        accuracy_test.append(accuracy_score(Y_true, Y_pred))

    accuracy_blcd_train_mean = np.mean(np.asarray(accuracy_blcd_train))
    accuracy_blcd_test_mean = np.mean(np.asarray(accuracy_blcd_test))
    accuracy_train_mean = np.mean(np.asarray(accuracy_train))
    accuracy_test_mean = np.mean(np.asarray(accuracy_test))

    return accuracy_blcd_train_mean, accuracy_blcd_test_mean, accuracy_train_mean, accuracy_test_mean, weights, data.columns

def get_most_relevant_features(weights, feat, num_feat_kept):
    # Determine the best parameters to use depending on the model
    # INPUT : - weights : list of arrays containing the weights associated to each features when fitting the model
    #         - feat : dataframe containing the features
    #         - num_feat_kept : percentage of features to keep
    # OUTPUT : - feat_relevant : arrays containing the features defined as most relevant the greatest number of times

    arr = np.zeros((len(feat)))
    dtf = pd.DataFrame([arr], columns=feat)
    print(weights)
    for i in range(len(weights)):
        k = 1
        weigths_ordered = np.argsort(weights[i])[::-1]
        for j in weigths_ordered:
            dtf[feat[j]] += k
            k += 1
    dtf = dtf.sort_values(by=[0], axis=1)
    print(dtf.iloc[0])
    feat_relevant = dtf.columns[0:num_feat_kept]
    return feat_relevant 

def stability_index(features_kept_table, N):
    # Return the index of stability 
    # INPUT : - features_kept_table : list of arrays containing the features defined as most relevant for each parameter tested
    #         - N : Total number of features
    # OUTPUT : - index : value of the index of stability

    features_kept = np.asarray(features_kept_table["indices"])
    index = 0
    tot = features_kept.shape[0]
    for k in range(len(features_kept)-1):
        for l in range(k+1,len(features_kept)):
            Si = features_kept[k]
            print(Si)
            Sj = features_kept[l]
            print(Sj)
            r = len(np.intersect1d(np.asarray(Si), np.asarray(Sj)))
            print(r)
            s = Si.shape[0]
            index += (r*N - s**2)/(s*(N-s)) 
            print(index)
    index = (2*index)/(tot*(tot-1))
    return index   

def weights_relevance(weights, feat):
    # Determine the best parameters to use depending on the model
    # INPUT : - weights : list of arrays containing the weights associated to each features when fitting the model
    #         - feat : dataframe containing the features
    # OUTPUT : - feat_relevant : DataFrame containing the features with a higher number for those which had a higher weight the greatest number of times

    arr = np.zeros((len(feat)))
    dtf = pd.DataFrame([arr], columns=feat)
    print(weights)
    for i in range(len(weights)):
        k = len(weights[0])
        weigths_ordered = np.argsort(weights[i])[::-1]
        for j in weigths_ordered:
            dtf[feat[j]] += k
            k = k-1
    dtf = dtf.sort_values(by=[0], axis=1)
    
    return dtf 

def get_weights_ordered(weights_dtf, param):
    weights_ordered = pd.DataFrame(columns=["Iteration", "Features_kept", "Importance", "names_feat"])
    for j in range(param.get("SFS")):
        for k in num_feats_keep:
            weights_toorder = (weights_dtf[(weights_dtf["Iteration"] == j) & (weights_dtf["Features_kept"] == k)]["Weights"]).to_list()
            labels = np.asarray(weights_dtf[(weights_dtf["Iteration"] == j) & (weights_dtf["Features_kept"] == k)]["names_feat"])[0]#[0]
            test_wgt = weights_relevance(weights_toorder, labels)
            print(np.asarray(test_wgt.iloc[0]))
            new_line_wgt = pd.DataFrame(data=np.array([[j, k, np.asarray(test_wgt.iloc[0]), test_wgt.columns]]), columns = ["Iteration", "Features_kept", "Importance", "names_feat"], dtype=object)
            weights_ordered = pd.concat([weights_ordered, new_line_wgt], ignore_index=True)
    return weights_ordered

def importance_weights(weights_ordered, num_feats_keep, param):
    col = (weights_ordered[(weights_ordered["Iteration"]==0) & (weights_ordered["Features_kept"]==num_feats_keep)]["names_feat"]).tolist()
    importance_final = pd.DataFrame(data = (weights_ordered[(weights_ordered["Iteration"]==0) & (weights_ordered["Features_kept"]==num_feats_keep)]["Importance"]).tolist(), columns= col)
    line_count = pd.DataFrame(data = [np.ones(len(np.asarray(col)[0]))], columns=col, dtype=object)
    importance_final = pd.concat([importance_final, line_count], ignore_index=True)
    for it in range(1,param.get("SFS")):
        names_toorder = (weights_ordered[(weights_ordered["Iteration"]==it) & (weights_ordered["Features_kept"]==num_feats_keep)]["names_feat"]).tolist()[0]
        for it_len in range(len(names_toorder)):
            if names_toorder[it_len] in importance_final.columns:
                importance_final.loc[[0], [names_toorder[it_len]]] += (weights_ordered[(weights_ordered["Iteration"]==it) & (weights_ordered["Features_kept"]==num_feats_keep)]["Importance"]).tolist()[0][it_len]
                importance_final.loc[[1], [names_toorder[it_len]]] += 1
            else:
                importance_final.insert(len(importance_final.columns.tolist()), names_toorder[it_len], [(weights_ordered[(weights_ordered["Iteration"]==it) & (weights_ordered["Features_kept"]==num_feats_keep)]["Importance"]).tolist()[0][it_len], 1])
                
    importance_final = importance_final.sort_values(by = [0], axis=1, ascending=False)            
    
    return importance_final
      
#---------------------------------------------------------------------------
# Code to run
#---------------------------------------------------------------------------

features = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_features/Features_table_nonnul_functionnel.csv")
features = features.drop("Unnamed: 0", axis=1)

for it in ['Three']:
    regions_kept = pd.read_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_corr_func_%s.p"%it)#pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/feature_selection2/regions_selected_corr_1.csv")
    features = features[regions_kept[0]].abs()
    features.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/features_kept_func_%s.csv"%(it,it))

    target = np.zeros(72)
    target[7:60] = 1
    target[60:72] = 2

    cor = np.corrcoef(features.T, target.T)
    cor_target = np.abs(cor[-1])
    indices = []
    for i in range(len(cor)-2):
        for j in range(i+1,len(cor)-1):
            if np.abs(cor[i,j]) > 0.8:
                if cor_target[i] > cor_target[j]:
                    indices.append(j)
                else:
                    indices.append(i)
    indices_sorted = sorted(list(set(indices)), reverse=True)
    print(len(indices_sorted))
    features = features.drop(features.columns[indices_sorted], axis=1)

    age = pd.read_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/patients_age.csv")
    age = age.drop("Unnamed: 0", axis=1)
    age = age.drop(index=[65], axis=0)
    age = age.drop(index=[19], axis=0)
    features.insert(0, "Age", age)

    features.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/features_kept_func_redundant_%s.csv"%(it,it))
    print(len(features.index))

    #Percentage of features selected
    num_feats_keep = (np.array([0.02, 0.04, 0.06, 0.08])*features.shape[1]).astype(int)

    #Find the best hyper parameters of the model
    param_rft = {'n_estimators': [10,50,100,150]}
    params_best = []
    count50 = 0
    count100 = 0
    count150 = 0
    count10 = 0
    for i in range(50):
        params_best_rft = find_best_params_rft(features, target, sklearn.ensemble.RandomForestClassifier(bootstrap=False), param_rft)
        params_best.append(params_best_rft)
        if params_best_rft.get("n_estimators") == 10:
            count10 += 1
        if params_best_rft.get("n_estimators") == 50:
            count50 += 1
        if params_best_rft.get("n_estimators") == 100:
            count100 += 1
        if params_best_rft.get("n_estimators") == 150:
            count150 += 1
    max = np.argmax([count10,count50,count100,count150])
    tab = [10,50,100,150]
    print("parameter chosen is", tab[max])

    #Sequential feature selection (wrapper method)
    param = {'SFS':35}
    r_list = Parallel(n_jobs=15)(delayed(machine_learning_func)(features, target, num_feats_keep, model_chosen = "rft", params = tab[max]) for i in range(param.get("SFS")))
    
    r_dtf = r_list[0]
    for il in range(1,len(r_list)):
        r_dtf = pd.concat([r_dtf, r_list[il]], ignore_index=True)
    print(r_dtf)
    r_dtf.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/r_dtf4_%s.p"%(it, it))

    stability = np.zeros(len(num_feats_keep))
    inc_m = 0
    for m in num_feats_keep:
        stability[inc_m] = stability_index(r_dtf[(r_dtf["num_features_kept"]==m)], len(features.columns))
        inc_m += 1   
    pd.DataFrame(stability).to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/stability4_%s.csv"%(it, it))

    df = pd.DataFrame(columns=["Iteration", "Features_kept", "Acc_balanced_train", "Acc_balanced_test","Acc_train","Acc_test"])
    weights_dtf = pd.DataFrame(columns=["Iteration", "Features_kept", "Weights", "names_feat"])
    for j in range(len(r_list)):
        for k in num_feats_keep:
            r1 = Parallel(n_jobs=5)(delayed(comparaison_accuracy)(features[features.columns[r_dtf[r_dtf["num_features_kept"]==k].iloc[j].indices]],target,'rft', tab[max]) for i in range(5))
            new_line = pd.DataFrame(data=np.array([[j,k,np.mean(np.array([r1[0][0], r1[1][0], r1[2][0], r1[3][0], r1[4][0]])),
                                                        np.mean(np.array([r1[0][1], r1[1][1], r1[2][1], r1[3][1], r1[4][1]])),
                                                         np.mean(np.array([r1[0][2], r1[1][2], r1[2][2], r1[3][2], r1[4][2]])),
                                                          np.mean(np.array([r1[0][3], r1[1][3], r1[2][3], r1[3][3], r1[4][3]]))]]),
                                                          columns=["Iteration", "Features_kept", "Acc_balanced_train", "Acc_balanced_test","Acc_train","Acc_test"],dtype=object)
            df = pd.concat([df,new_line],ignore_index=True)
            
            
            for l in range(len(r1)):
                for m in range(5): #Because kfold of 5 in comparaison_accuracy
                    new_line_weight = pd.DataFrame(data=np.array([[j, k, r1[l][4][m], r1[l][5]]]), columns=["Iteration", "Features_kept", "Weights", "names_feat"],dtype=object)
                    weights_dtf = pd.concat([weights_dtf,new_line_weight],ignore_index=True)     
    
    df.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/accuracy4_%s.csv"%(it, it))

    weights_dtf.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/weigts4_%s.csv"%(it, it))
    weights_dtf.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/weigts4_%s.p"%(it, it))

    weights_ordered = get_weights_ordered(weights_dtf, param)

    weights_ordered.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/weights_ordered4_%s.csv"%(it, it))
    weights_ordered.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/weights_ordered4_%s.p"%(it, it))

    final_importance_001 = importance_weights(weights_ordered, num_feats_keep[0], param)
    final_importance_002 = importance_weights(weights_ordered, num_feats_keep[1], param)
    final_importance_003 = importance_weights(weights_ordered, num_feats_keep[2], param)
    final_importance_004 = importance_weights(weights_ordered, num_feats_keep[3], param)
    
    final_importance_001.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_002_%s.csv"%(it, it))
    final_importance_001.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_002_%s.p"%(it, it))
    final_importance_002.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_004_%s.csv"%(it, it))
    final_importance_002.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_004_%s.p"%(it, it))
    final_importance_003.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_006_%s.csv"%(it, it))
    final_importance_003.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_006_%s.p"%(it, it))
    final_importance_004.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_008_%s.csv"%(it, it))
    final_importance_004.to_pickle("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/final_importance4_008_%s.p"%(it, it))

    mean_per_num_feat_kept = pd.DataFrame(columns=["Mean Acc_balanced_train", "Mean Acc_balanced_test", "Mean Acc_train", "Mean Acc_test"])
    for num in num_feats_keep:
        new_line_mean = pd.DataFrame([[np.mean(np.asarray(df[df["Features_kept"]==num]['Acc_balanced_train'])), 
                                        np.mean(np.asarray(df[df["Features_kept"]==num]['Acc_balanced_test'])), 
                                        np.mean(np.asarray(df[df["Features_kept"]==num]['Acc_train'])), 
                                        np.mean(np.asarray(df[df["Features_kept"]==num]['Acc_test']))]],
                                        columns=["Mean Acc_balanced_train", "Mean Acc_balanced_test", "Mean Acc_train", "Mean Acc_test"])
        mean_per_num_feat_kept = pd.concat([mean_per_num_feat_kept, new_line_mean], ignore_index=True)
    
    mean_per_num_feat_kept.to_csv("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/wrapped_features_selection/%s/func/rft/mean_accuracy4_%s.csv"%(it, it))

    
    

    
