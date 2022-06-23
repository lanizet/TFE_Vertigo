# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:29:17 2022

@author: anizetlaura & doyenastrid
"""

# Import of packages
#---------------------------------------------------------------------------

import numpy as np
from scipy.ndimage.morphology import binary_dilation
from dipy.data import default_sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
from dipy.direction import peaks
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion, ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
import pandas as pd
import json
import datetime
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, load_trk
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm
from dipy.direction import ProbabilisticDirectionGetter
from dipy.data import small_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from dipy.tracking.utils import length
import pickle


# Definition of functions
#---------------------------------------------------------------------------
def clean_tracto(streamlines, params, patients):
    '''

    Parameters
    ----------
    streamlines : array containing streamlines
    threshold : length below which streamlines are removed

    Returns
    -------
    new_streamlines : array of streamlines with length superior to threshold

    '''
    
    data, data_affine, data_img = load_nifti(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(patients,patients), return_img=True)
    
    longueur = streamlines._lengths

    index = np.argwhere(longueur < params.get('threshold'))
    new_tracto = np.delete(streamlines,index)
    
    sft1 = StatefulTractogram(new_tracto, data_img, Space.RASMM)
    save_trk(sft1, path+"structural_tractogramme/%s/%s_tractogram_struct_proba_cleaned_%s.trk"%(patients,patients,params.get('it')))
    
    return sft1, new_tracto


def erode3D(ROI,repeat=1):
    '''
    

    Parameters
    ----------
    ROI : 3D binary mask    
    repeat : numbers of times the operation is repeated, default=1

    Returns
    -------
    ROI : 3D binary mask eroded by a 3-wide cross kernel

    '''

    while repeat>0:
        ROI_copy=ROI.copy()
        
        for i in range(len(ROI.shape)):
        
            orderLeft=list(range(1,ROI.shape[i]))+[0]
            orderRight=[ROI.shape[i]-1]+list(range(0,ROI.shape[i]-1))
            
            for order in [orderLeft,orderRight]:
            
                if i==0:
                    ROI+=ROI_copy[order,:,:]
                elif i==1:
                    ROI+=ROI_copy[:,order,:]
                elif i==2:
                    ROI+=ROI_copy[:,:,order]
        
        ROI[ROI<7]=0
        ROI[ROI>=7]=1
        repeat-=1
    
    return ROI

# Code to run
#---------------------------------------------------------------------------
def Tractogram_proba(patients, params):
    '''
    Function building and saving the tractogram for a patient.

    Parameters
    ----------
    patients : patient for which the tractogram must be computed
    params : discioary of parameters that are used to build the tractogram. The dictionary must be under the form params={'fa_threshold':.4,'gfa_threshold':.25,'max_angle':20,'step_size':1,'density':3, 'repeat': 2, 'threshold': 10, 'it': 8}

    
    '''

    #Write in a json file the parameters used to build the tratogram
    with open(path+"structural_tractogramme/%s/params_proba_%s.json"%(patients,patients), "a") as file:
        json.dump(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"), file)
        json.dump(params, file)
        file.write("\n")

    #Load the preprocessed diffusion images, the b-values and b-vectors, the brain mask and the atlas
    data, data_affine, data_img = load_nifti(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(patients,patients), return_img=True)
    bvals, bvecs = read_bvals_bvecs(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/dMRI/preproc/%s_dmri_preproc.bval"%(patients,patients), path+"ELIKOPY_subset_new/PROJECT/subjects/%s/dMRI/preproc/%s_dmri_preproc.bvec"%(patients,patients))
    gtab = gradient_table(bvals, bvecs)
    labels = load_nifti_data(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/masks/%s_brain_mask.nii.gz"%(patients,patients))
    atlas_final = load_nifti_data(path+"/Atlas/%s/%s_registered_atlas_FA.nii.gz"%(patients,patients))
    
    #The brain mask is eroded and used as seed points
    affine = data_affine
    white_matter = erode3D(labels,repeat=params.get("repeat"))
    seeds = utils.seeds_from_mask(white_matter, affine, density=params.get("density"))
    
    #Computation of the model representing the orientation of fibers
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=params.get("fa_threshold"))
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(data, mask=white_matter)
    
    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(data, mask=white_matter).gfa
    stopping_criterion = ThresholdStoppingCriterion(gfa, params.get("gfa_threshold"))
    
    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                         max_angle=params.get("max_angle"),
                                                         sphere=default_sphere)
 
    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                             affine, step_size=params.get("step_size"))
    streamlines = Streamlines(streamline_generator)
    sft = StatefulTractogram(streamlines, data_img, Space.RASMM)
    #Save the obtained tractogram as trk file
    save_trk(sft, path+"structural_tractogramme/%s/%s_tractogram_struct_proba_%s.trk"%(patients,patients,params.get("it")))
    
    #Save the corresponding streamlines as pickle file
    with open(path+"structural_tractogramme/%s/%s_streamlines_%s.p"%(patients,patients,params.get("it")), 'wb') as f1:
         pickle.dump(streamlines, f1)
    
    #Clean the tractogram to remove streamlines having a length  below threshold
    sft2, new_streamlines = clean_tracto(streamlines,params,patients)


    #CONNECTIVITY -------------------------------------------------------------
    #Computation of the connectivity matrix based on the streamlines and using the atlas
    M, grouping = utils.connectivity_matrix(new_streamlines, affine,
                                            atlas_final.astype(np.uint32),
                                            return_mapping=True,
                                            mapping_as_streamlines=True,
                                            inclusive=False)
    
    #Load the regions of the atlas non registered on the diffusion image
    with open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_registration/sorted_nonregisteredlabels.json") as file:
        labels_nonregistered = json.load(file)
    
    #Suppression of the non registered regions from the connectivity matrix
    new_M = np.zeros((281,281))
    new_M[0:len(M),0:len(M[0])] = M

    conn_matrix = np.delete(new_M, labels_nonregistered, 0)
    conn_matrix = np.delete(conn_matrix, labels_nonregistered, 1)

    #Save the raw connectivity matrix
    np.savetxt(path+"structural_connectome/%s/connectivity2_%s.txt"%(patients,patients), conn_matrix, fmt='%f')

    #Show the raw connectivity matrix
    plt.figure()
    plt.imshow(np.log1p(conn_matrix), interpolation='nearest')
    plt.savefig(path+"structural_connectome/%s/connectivity2_%s.png"%(patients,patients))
    plt.show()


    #Ponderation of the connectivity matrix w/ surface to take into account the difference in surfaces of the different regions
    # ------------------------------------
    volume_matrix = np.zeros(281)
    regionListFromAtlas = np.arange(0,281,1)
    for i in (regionListFromAtlas):
        volume_matrix[i] = len(np.where(atlas_final==i)[0])

    ponderation_matrix = np.zeros((281,281))
    for i in range(0,281):
        for j in range(0,281):
            ponderation_matrix[i,j] = volume_matrix[i] + volume_matrix[j]


    pond_matrix = np.delete(ponderation_matrix, labels_nonregistered, 0)
    pond_matrix = np.delete(pond_matrix, labels_nonregistered, 1)
    np.fill_diagonal(conn_matrix,0)
    connectivity_matrix_pondW = np.divide(conn_matrix, pond_matrix)

    #Save the connectivity matrix corrected for the surface
    np.savetxt(path+"structural_connectome/%s/connectivity_pondW2_%s.txt"%(patients,patients), connectivity_matrix_pondW, fmt='%f')

    #Show the connectivity matrix corrected for the surface
    plt.figure()
    plt.imshow(np.log1p(connectivity_matrix_pondW), interpolation='nearest')
    plt.savefig(path+"structural_connectome/%s/connectivity_pondW2_%s.png"%(patients,patients))
    plt.show()

    #Ponderation of the connectivity matrix  w/ length------------------------------------
    longueur = np.zeros((281,281))
    rangefor = np.arange(0,281)
    for i in (rangefor):
        for j in (rangefor[i::]): 
            sum = 0
            for k in range(0,len(grouping[i,j])):
                sum += len(grouping[i,j][k])
            if (len(grouping[i,j])!=0):
                longueur[i,j] = sum/len(grouping[i,j])
                longueur[j,i] = sum/len(grouping[i,j])
    new_longueur = np.delete(longueur, labels_nonregistered, 0)
    new_longueur_bis = np.delete(new_longueur, labels_nonregistered, 1)
    np.fill_diagonal(new_longueur_bis,0)

    new_longueur_bis[np.where(new_longueur_bis==0)] = 1

    connectivity_matrix_pondL = np.divide(conn_matrix, new_longueur_bis)

    #Save the connectivity matrix corrected for the length of streamlines
    np.savetxt(path+"structural_connectome/%s/connectivity_pondL2_%s.txt"%(patients,patients), connectivity_matrix_pondL, fmt='%f')

    #Show the connectivity matrix corrected for the length of streamlines
    plt.figure()
    plt.imshow(np.log1p(connectivity_matrix_pondL), interpolation='nearest')
    plt.savefig(path+"structural_connectome/%s/connectivity_pondL2_%s.png"%(patients,patients))
    plt.show()

    #Ponderation of the connectivity matrix  w/ length & weight------------------------------------
    connectivity_matrix_pond = np.divide(connectivity_matrix_pondW, new_longueur_bis)
    connectivity_matrix_pond = connectivity_matrix_pond*100000

    #Save the connectivity matrix corrected for the length of streamlines and surface
    np.savetxt(path+"structural_connectome/%s/connectivity_pond2_%s.txt"%(patients,patients), connectivity_matrix_pond, fmt='%f')

    #Show the connectivity matrix corrected for the length of streamlines and surface
    plt.figure()
    plt.imshow(np.log1p(connectivity_matrix_pond), interpolation='nearest')
    plt.savefig(path+"structural_connectome/%s/connectivity_pond2_%s.png"%(patients,patients))
    plt.show() 
    
    print("%s done"%patients)

#CODE TO RUN
#------------------------------------------------------------------------
patient_list = ["C_"+str(_) for _ in range(5,13)] + ["H_"+str(_) for _ in range(7)]#["V_"+str(_) for _ in range(0,12)] + ["C_"+str(_) for _ in range(0,4)] + #["H_"+str(_) for _ in range(7)] #["V_"+str(_) for _ in range(32,54)] + ["C_"+str(_) for _ in range(0,4)] + ["C_"+str(_) for _ in range(5,10)] + ["C_"+str(_) for _ in range(11,13)] # + ["V_"+str(_) for _ in range(12)] + ["V_"+str(_) for _ in range(13,31)] + ["V_"+str(_) for _ in range(32,54)] + ["C_"+str(_) for _ in range(0,4)] + ["C_"+str(_) for _ in range(5,10)] + ["C_"+str(_) for _ in range(11,13)]
path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"

# Tractogram + connectivity matrix -------------------------------------------------------------
params={'fa_threshold':.4,'gfa_threshold':.25,'max_angle':20,'step_size':1,'density':3, 'repeat': 2, 'threshold': 10, 'it': 8}

Parallel(n_jobs=10)(delayed(Tractogram_proba)(patient_list[i], params) for i in range(len(patient_list)))