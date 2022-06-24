# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:40:14 2022

@author: anizetlaura & doyenastrid
"""

#---------------------------------------------------------------------------
# Import of packages
#---------------------------------------------------------------------------
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import plotting
import json
from joblib import Parallel, delayed
import nibabel as nib

#---------------------------------------------------------------------------
# Function definition
#---------------------------------------------------------------------------
def connectivity_functionel(patients):
        masker = NiftiLabelsMasker(labels_img= path+"functional_registration/%s/%s_registered_atlas_MNItoT1_orfunc.nii.gz"%(patients,patients), standardize=True, high_variance_confounds=True)

        time_series = masker.fit_transform(path+"functional_registration/%s/%s_registered_functoT1.nii.gz"%(patients,patients))

        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]

        np.fill_diagonal(correlation_matrix, 0)
        # The labels we have start with the background (0), hence we skip the
        # first label
        # matrices are ordered for block-like representation
        display = plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=np.arange(1,len(correlation_matrix)+1,1),
                             vmax=0.8, vmin=-0.8, reorder=False, cmap = 'viridis')
        
        display.figure.savefig(path+"functional_connectome/%s/func_connectome_%s.png"%(patients,patients))
        np.savetxt(path+"functional_connectome/%s/func_connectome_%s.txt"%(patients,patients), correlation_matrix, fmt='%f')

        print("Connectivity for patient %s done"%patients)

        registered_label = masker.labels_img_
        labels = registered_label._data_cache

        functional_img = nib.load(path+"functional_registration/%s/%s_registered_functoT1.nii.gz"%(patients,patients)).get_fdata()
        slice0 = functional_img[:,:,:,0]
        nul_values = np.argwhere(slice0 == 0)

        labels[nul_values[:,0],nul_values[:,1],nul_values[:,2]] = 0

        atlas_affine = nib.load(path+"functional_registration/%s/%s_registered_atlas_MNItoT1_orfunc.nii.gz"%(patients,patients)).affine
        out=nib.Nifti1Image(labels,atlas_affine)
        out.to_filename(path+"functional_connectome/%s/%s_labels_onfunc.nii.gz"%(patients,patients))

        nonRegistered = []
        for j in range (0, 281):
            if(np.count_nonzero(labels==j)==0):
                nonRegistered.append(j)

        with open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_connectome/registration_functionnel.json", "a") as file:
             json.dump(patients, file)
             json.dump(nonRegistered, file)
             file.write("\n")

#---------------------------------------------------------------------------
# Code to run
#---------------------------------------------------------------------------
if __name__=='__main__':

    patient_list = ["H_"+str(_) for _ in range(7)] + ["V_"+str(_) for _ in range(0,12)] + ["V_"+str(_) for _ in range(13,54)] + ["C_"+str(_) for _ in range(13)]  
    
    path='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/'
    
    Parallel(n_jobs=10)(delayed(connectivity_functionel)(patient_list[i]) for i in range(len(patient_list)))