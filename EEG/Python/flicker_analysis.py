from mat2python import getFreqValuesVec, folderIterator, chanIndIterator
import numpy as np
import os

def getFlickers(iter_list):
    flicker_type = []
    flicker_powers=[]
    for itr in iter_list:
        power_ppt = []
        for file_path in itr:
            power,spectrum=getFreqValuesVec(file_path)
            pat_num = int(os.path.basename(file_path)[-7:-4])# Getting the ppt number 
            if (pat_num % 2) == 0: 
                flickers = [6,12,15,30]
                flicker_type.append(0)
            else:
                flickers = [7.5,15,12,24]
                flicker_type.append(1)

            for f in flickers:
                ind_peak = spectrum.index(f)
                power_ppt.append(power[ind_peak])
        flicker_powers.append(power_ppt)
    return flicker_powers,flicker_type



fooof_vars_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/fooof_fit_values'
## off - on
freq_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/FreqValuesConds/'
loc_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/chanLocs/'
conds = ['Off','On','MW','MB']

folder_iter_wrapper = lambda cond: folderIterator(freq_path + '/'+cond)

iter_list = map(folder_iter_wrapper,conds)

flicker_powers,flicker_type = getFlickers(iter_list)



