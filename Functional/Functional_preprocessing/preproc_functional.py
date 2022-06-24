# -*- coding: utf-8 -*-
"""

@author: Anizet Laura & Doyen Astrid

"""

import os
import json

root='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/'
smoothing = True

#List of patients
PatientList=["C_0", "C_1", "C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "C_9", "C_10", "C_11", "C_12", "H_0", 
                 "H_1", "H_2", "H_3", "H_4", "H_5", "H_6", "V_0", "V_1", "V_2", "V_3", "V_4", "V_5",
                 "V_6", "V_7", "V_8", "V_9", "V_10", "V_11", "V_13", "V_14", "V_15", "V_16", "V_17", 
                 "V_18", "V_19", "V_20", "V_21", "V_22", "V_23", "V_24", "V_25", "V_26", "V_27", "V_28", "V_29",
                 "V_30", "V_31", "V_32", "V_33", "V_34", "V_35", "V_36", "V_37", "V_38", "V_39", "V_40", "V_41",
                 "V_42", "V_43", "V_44", "V_45", "V_46", "V_47", "V_48", "V_49", "V_50", "V_51", "V_52", "V_53"]


for Patient in PatientList:
    if smoothing == True:
        os.system('sbatch /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_preprocessing/preproc_functional_smooth.sh '+Patient)

    else:
        os.system('sbatch /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_preprocessing/preproc_functional.sh '+Patient)
