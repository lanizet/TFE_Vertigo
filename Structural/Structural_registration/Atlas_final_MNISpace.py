#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 21:04:35 2022

@author: anizetlaura & doyenastrid
"""

#The goal of this script is to put the two altases, Brainnetome and SUIT in one atlas in the MNI space.

# Import of packages
#---------------------------------------------------------------------------

import numpy as np
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
import pandas as pd

#---------------------------------------------------------------------------



# Definition of functions
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

def applyTransform(moving_data,mapping,static,static_affine,register=False,output_path=None,binary=False):
    
    '''

    Parameters
    ----------
    static : nib.load(static_file).get_fdata()
    moving : nib.load(moving_file_path).get_fdata()
    register : if True, register the transformed image in the output path

    '''  

    transformed=mapping.transform(moving_data)
    
    if binary:
        transformed[transformed>.5]=1
        transformed[transformed<=.5]=0
    
    if register:
        out=nib.Nifti1Image(transformed,static_affine)#,header=static.header)
        out.to_filename(output_path)
  
    else:
        return transformed
 

# Code to run
#---------------------------------------------------------------------------

if __name__=='__main__':

    #Brainnetome atlas
    bn = nib.load("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/BN_Atlas_246_1mm.nii")
    bn_img = bn.get_fdata()
    bn_affine = bn.affine
    
    #SUIT atlas
    suit = nib.load("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/atl-Anatom_space-MNI_dseg.nii")
    suit_img = suit.get_fdata()
    suit_affine = suit.affine
    
    dmap = getTransform(bn_img, bn_affine, suit_img, suit_affine, onlyAffine=True)
   
    new_atlas = bn_img.copy()
    registered_atlas = np.zeros(bn_img.shape)
    
    regionListFromAtlas = np.arange(1,35,1)
    
    for label_of_region in tqdm(regionListFromAtlas):
        region_atlas = suit_img.copy()
        region_atlas[region_atlas!=label_of_region] = 0
        region_atlas[region_atlas==label_of_region] = 1
        out = nib.Nifti1Image(region_atlas, suit_affine)
        out.to_filename("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/Regions_atlas_cervelet/"+str(label_of_region)+".nii.gz")
        
        registered_region=applyTransform(region_atlas, dmap, bn_img, bn_affine)
        registered_atlas[registered_region > 0.5]=label_of_region
    out=nib.Nifti1Image(registered_atlas,bn_affine)
    out.to_filename("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/registered_atlas_cervelet.nii.gz")
    
    for i in range (0, 35):
        if(np.count_nonzero(registered_atlas==i)==0):
            print(i, "fail d'atlas suit")
            
    suit_MNI_scaled = np.asarray(registered_atlas, dtype = np.int32)
    suit_MNI_scaled+=246
    suit_MNI_scaled[suit_MNI_scaled==246]=0
    suit_MNI_scaled[np.where((new_atlas != 0) & (suit_MNI_scaled != 0))] = 0 #200 pixels ou ca s'overlap
    new_atlas+=suit_MNI_scaled

    
    out = nib.Nifti1Image(new_atlas, bn_affine)
    out.to_filename("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/atlas_bn_suit.nii.gz") #Notre atlas final !! (normalement)


    #Manipulation of the atlas to have first left en then right regions
    for i in range (0, 281):
        if(np.count_nonzero(new_atlas==i)==0):
            print(i, "fail d'atlas final")
            
    atlas_lr = np.zeros_like(new_atlas)  
    k = 1
    for i in range (1, 247, 2):
        index = np.where(new_atlas == i)
        atlas_lr[index] = k
        k = k+1
        

    k = 137
    for i in range (2, 247, 2):
        index = np.where(new_atlas == i)
        atlas_lr[index] = k
        k = k+1
        
    excel_suit = pd.read_excel(r"/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/Labels_AtlasSuit.xlsx")
    labels_suit = excel_suit["name"]
    print(len(labels_suit))
    
    k = 124
    for i in range (0, len(labels_suit)):
        if "Left" in labels_suit[i].split("_"):
            index_cervelet_left = excel_suit["index"][i]
            index = np.where(new_atlas == index_cervelet_left)
            atlas_lr[index] = k
            k = k+1
    
    k = 260
    for i in range (0, len(labels_suit)):
        if "Right" in labels_suit[i].split("_"):
            index_cervelet_right = excel_suit["index"][i]
            index = np.where(new_atlas == index_cervelet_right)
            atlas_lr[index] = k
            k = k+1

    k = 273
    for i in range (0, len(labels_suit)):
        if "Vermis" in labels_suit[i].split("_"):
            index_cervelet_vermis = excel_suit["index"][i]
            index = np.where(new_atlas == index_cervelet_vermis)
            atlas_lr[index] = k
            k = k+1

    out = nib.Nifti1Image(atlas_lr, bn_affine)
    out.to_filename("/Users/anizetlaura/Documents/Unif/MASTER 2/Memoire/CODES/Atlas/new_atlas.nii.gz")
    