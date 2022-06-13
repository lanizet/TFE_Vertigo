import elikopy
import elikopy.utils
import os
     
#ElikoPy is used for the preprocessing of structural data and this code was used on the CECI clusters

#Path containing the raw dara
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/"  
 
study = elikopy.core.Elikopy(f_path,slurm=True, slurm_email="astrid.doyen@student.uclouvain.be")
patient_list= ["H_"+str(_) for _ in range(0,7)] + ["V_"+str(_) for _ in range(0,54)] + ["C_"+str(_) for _ in range(0,13)]

#For the rest of the code, it is runed by decommenting one line at a time

#study.patient_list()

#study.preproc(reslice=False, denoising=True, gibbs=False, topup=True, forceSynb0DisCo=False, useGPUsynb0DisCo=False, eddy=True, biasfield=True, patient_list_m=patient_list, starting_state=None, bet_dilate=2, slurm=None, slurm_timeout="5:00:00", cpus=3, slurm_mem=6096, qc_reg=False, report=True, **preproc_kwargs)

#study.white_mask(patient_list_m=patient_list,**wm_kwargs)
 
#study.dti(patient_list_m=patient_list, use_wm_mask=False) # patient_list_m=patient_list, use_wm_mask=False