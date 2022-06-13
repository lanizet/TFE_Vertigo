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
from dipy.io.image import save_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import nibabel as nib
from tqdm import tqdm
from Atlas_final_MNISpace import getTransform, applyTransform
import json 

# Code to run
#---------------------------------------------------------------------------

if __name__=='__main__':
    
    patients = ["H_"+str(_) for _ in range(7)] + ["V_"+str(_) for _ in range(0,3)] + ["V_"+str(_) for _ in range(4,12)] + ["V_"+str(_) for _ in range(13,54)] + ["C_"+str(_) for _ in range(1,13)]
    
    path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"

    #Upload FA map in the MNI space
    img_moving = nib.load(path+"Atlas/FSL_HCP1065_FA_1mm.nii.gz")
    FA_moving = img_moving.get_fdata()
    FA_moving_affine = img_moving.affine
    
    #Upload atlas in the MNI space
    regionListFromAtlas = np.arange(1,281,1)
    atlas_final = nib.load(path+"Atlas/new_atlas.nii.gz").get_fdata()
    
    nonRegisteredRegions_table = []
    
    #Registration for each patient
    for i in range(0,len(patients)):
        FA = nib.load(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/dMRI/microstructure/dti/%s_FA.nii.gz"%(patients[i],patients[i]))
        FA_static = FA.get_fdata()
        FA_static_affine = FA.affine
        
        #Get the transformation between the native space of each patient and the MNI space
        dmap_fa=getTransform(FA_static, FA_static_affine, FA_moving, FA_moving_affine, diffeomorph=True) #Mettre un jour diffemorph=True pour avoir meilleurs resultats
        
        registered_atlas = np.zeros(FA_static.shape)

        #Apply the registration for each region separately
        for label_of_region in regionListFromAtlas:
            region_atlas=atlas_final.copy()
            region_atlas[region_atlas!=label_of_region]=0
            region_atlas[region_atlas==label_of_region]=1
            out = nib.Nifti1Image(region_atlas, FA_moving_affine)
            out.to_filename(path+"Atlas/%s/Regions_atlas_final/"%patients[i]+str(label_of_region)+".nii.gz")
            
            registered_region=applyTransform(region_atlas, dmap_fa, FA_static, FA_static_affine)
            
            registered_region[registered_region > 0.5] = label_of_region
            registered_region[registered_region <= 0.5] = 0
            out = nib.Nifti1Image(registered_region, FA_moving_affine)
            out.to_filename(path+"Atlas/%s/Regions_atlas_final_registered/"%patients[i]+str(label_of_region)+".nii.gz")
            
        out=nib.Nifti1Image(registered_atlas,FA_static_affine)
        out.to_filename(path+"Atlas/%s/%s_registered_atlas_FA.nii.gz"%(patients[i],patients[i]))

        #Identify the regions non registered during the process (basically, the cerebellum)
        nonRegistered = []
        for j in range (0, 281):
            if(np.count_nonzero(registered_atlas==j)==0):
                nonRegistered.append(j)
        nonRegisteredRegions_table.append(nonRegistered)

        #json file gathering for each patient the non registered regions
        with open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_registration/registration.json", "a") as file:
            json.dump(patients[i], file)
            json.dump(nonRegistered, file)
            file.write("\n")

#json file gathering a big list with all the non registered regions
with open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/structural_registration/registration_table.json", "a") as file:
    json.dump(nonRegisteredRegions_table, file)
    


   
    