# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:40:14 2022

@author: anizetlaura & doyenastrid
"""

#---------------------------------------------------------------------------

# Import packages
#---------------------------------------------------------------------------

import numpy as np
import nibabel as nib
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import json
from joblib import Parallel, delayed


#---------------------------------------------------------------------------

# Function definition
#---------------------------------------------------------------------------

def getTransform(static,static_affine,moving,moving_affine,onlyAffine=False,
                 diffeomorph=False):
    '''

    Parameters
    ----------
    static : 3D array of static volume (nib.load(static_volume_file).get_fdata())
    static_affine : nib.load(static_volume_file).affine
    moving : 3D array of moving volume(nib.load(moving_volume_file).get_fdata())
    moving_affine : nib.load(moving_volume_file).affine
    diffeomorph : if False then registration is only affine
    onlyAffine : If registration of an atlas to a brain's patient : False
                 If registration on the MNI space : True

    Returns
    -------
    mapping : transform operation to send moving_volume to static_volume space

    '''    

    if len(static.shape) == 4:
        static = static[:,:,:,0]
    static_grid2world = static_affine
    
    moving_grid2world = moving_affine
    
    # Affine registration -----------------------------------------------------
    
    if onlyAffine:
        
        identity = np.eye(4)
        affine_map = AffineMap(identity,
                               static.shape, static_grid2world,
                               moving.shape, moving_grid2world)
        
        return affine_map
    
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=c_of_mass.affine)
    
    transform = RigidTransform3D()
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)
    
    transform = AffineTransform3D()
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=rigid.affine)
    
    # Diffeomorphic registration --------------------------
    
    if diffeomorph:
    
        metric = CCMetric(3)
        
        level_iters = [10000, 1000, 100]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        
        mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                               affine.affine)
        
    else:
        
        mapping = affine
    
    return mapping

def applyTransform(moving_data,mapping,static,static_affine,fct_runned,register=False,output_path=None,binary=False): 
    '''

    Parameters
    ----------
    static : nib.load(static_file).get_fdata()
    moving : nib.load(moving_file_path).get_fdata()
    register : if True, register the transformed image in the output path

    '''  
    if fct_runned != 'func_MNItoPat':
        transformed = np.zeros((static.shape[0],static.shape[1],static.shape[2],180))
        for i in range(180):
            transformed[:,:,:,i]=mapping.transform(moving_data[:,:,:,i])
    
    else:
        transformed = mapping.transform(moving_data)
    
    if binary:
        transformed[transformed>.5]=1
        transformed[transformed<=.5]=0
    
    if register:
        out=nib.Nifti1Image(transformed,static_affine)#,header=static.header)
        out.to_filename(output_path)
  
    else:
        return transformed
    

def func_MNItoPat(patients):
    #--------MNIToPatientSpace

    img_moving = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/Atlas/mni.nii.gz")
    T1_moving = img_moving.get_fdata()
    T1_moving_affine = img_moving.affine

    static_img = nib.load(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/T1/%s_T1_corrected_extracted.nii.gz"%(patients,patients))
    T1_static = static_img.get_fdata()
    T1_static_affine = static_img.affine

    dmap_t1=getTransform(T1_static, T1_static_affine, T1_moving, T1_moving_affine, diffeomorph=True) #Mettre un jour diffemorph=True pour avoir meilleurs resultats
    
    regionListFromAtlas = np.arange(1,281,1)
    atlas_final = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/Atlas/new_atlas.nii.gz").get_fdata()
        
    registered_atlas = np.zeros(T1_static.shape)
    
    for label_of_region in (regionListFromAtlas):
        region_atlas=atlas_final.copy()
        region_atlas[region_atlas!=label_of_region]=0
        region_atlas[region_atlas==label_of_region]=1
        out = nib.Nifti1Image(region_atlas, T1_moving_affine)
        out.to_filename("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_registration/%s/Regions_atlas_final/"%patients+str(label_of_region)+".nii.gz")

        region_atlas = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_registration/%s/Regions_atlas_final/"%patients+str(label_of_region)+".nii.gz").get_fdata()

        registered_region=applyTransform(region_atlas, dmap_t1, T1_static, T1_static_affine, fct_runned='func_MNItoPat')
        registered_atlas[registered_region > 0.5]=label_of_region
    out=nib.Nifti1Image(registered_atlas,T1_static_affine)
    out.to_filename(path+"functional_registration/%s/%s_registered_atlas_MNItoT1_orfunc.nii.gz"%(patients,patients))

     
def func_PattoMNI(patients) :   
    #--------PatientToMNISpace
    img_staticb = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/Atlas/mni.nii.gz")
    T1_staticb = img_staticb.get_fdata()
    T1_static_affineb = img_staticb.affine

    for i in range(0,len(patients)):
        img_movingb = nib.load(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/T1/%s_T1_corrected_extracted.nii.gz"%(patients,patients))
        T1_movingb = img_movingb.get_fdata()
        T1_moving_affineb = img_movingb.affine
        
        dmap_t1b=getTransform(T1_staticb, T1_static_affineb, T1_movingb, T1_moving_affineb,diffeomorph=True)
        
        atlas_img = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/Atlas/new_atlas.nii.gz")
        atlas_final_data = atlas_img.get_fdata()
        atlas_final_affine = atlas_img.affine
        
        fMRI = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/Atlas/%s/%s_registered_functoT1.nii.gz"%(patients,patients)).get_fdata()
        
        registered_toMNI=applyTransform(fMRI, dmap_t1b, T1_staticb, T1_static_affineb, fct_runned='func_PattoMNI')
        out=nib.Nifti1Image(registered_toMNI,T1_static_affineb)
        out.to_filename("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_registration/%s/%s_registered_functoMNI.nii.gz"%(patients,patients))
    
def funcToT1(patients):
    T1_static = nib.load(path+"ELIKOPY_subset_new/PROJECT/subjects/%s/T1/%s_T1_corrected_extracted.nii.gz"%(patients,patients))
    T1_staticb = T1_static.get_fdata()
    T1_static_affineb = T1_static.affine
    fMRI_moving = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/fMRI/preproc_nosmooth/mc_%s/highpass/highpass_%s.nii.gz"%(patients,patients))
    fMRI_movingb = fMRI_moving.get_fdata()
    fMRI_moving_affineb = fMRI_moving.affine
    dmap = getTransform(T1_staticb, T1_static_affineb, fMRI_movingb, fMRI_moving_affineb, diffeomorph=True, onlyAffine=True)
    registered_fMRI = applyTransform(fMRI_movingb, dmap, T1_staticb, T1_static_affineb, fct_runned='funcToT1')
    out=nib.Nifti1Image(registered_fMRI,T1_static_affineb)
    out.to_filename("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_registration/%s/%s_registered_functoT1.nii.gz"%(patients,patients))
    
#---------------------------------------------------------------------------
# Code to run
#---------------------------------------------------------------------------

if __name__=='__main__':
    
    patient_list = ["H_"+str(_) for _ in range(7)] + ["V_"+str(_) for _ in range(0,12)] + ["V_"+str(_) for _ in range(13,54)] + ["C_"+str(_) for _ in range(13)] 
    path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/"
    
    Parallel(n_jobs=10)(delayed(funcToT1)(patient_list[i]) for i in range(len(patient_list)))
    
    Parallel(n_jobs=10)(delayed(func_MNItoPat)(patient_list[i]) for i in range(len(patient_list)))
    
    Parallel(n_jobs=5)(delayed(func_PattoMNI)(patient_list[i]) for i in range(len(patient_list)))
    