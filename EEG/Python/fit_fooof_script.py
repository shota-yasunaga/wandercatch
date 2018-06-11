import os 
from fooof import FOOOF
from mat2python import getFreqValuesVec,getChanLocs,folderIterator,chanIndIterator
import numpy as np
from util import cd

def save_file(file_name,variable):
    print()
    f = open(file_name,"w+")
    np.save(file_name,variable)
    f.close()

def run_fooof(freq_path,loc_path,save_report_path,save_variables_path,save_variable_prefix,save_variables = True
,save_report=True,freq_range=[1,30],channel = 'Oz'):

    freqIterator = folderIterator(freq_path)
    chan_inds_iterator = chanIndIterator(loc_path)

    # Constant
    bg_ppts= []# background parameters/ppts
    peak_ppts = []
    errors = []
    for chan_ind, freq_var in zip(chan_inds_iterator,freqIterator)
        if True: # if files[0].endswith(".mat"): TODO: set the conditions for safety            
            power,spectrum = getFreqValuesVec(freq_var)
            if chan_ind != -1
                power = 10**np.array(power[chan_ind])
                spectrum=np.array(spectrum).flatten()
                # Initialize FOOOF object
                fm = FOOOF(peak_width_limits=[0.5,8])
                
                # Model the power spectrum with FOOOF, and print out a report
                fm.fit(spectrum, power, freq_range)

                bg_ppts.append(fm.background_params_)
                peak_ppts.append(fm.peak_params_)
                errors.append(fm.error_)
                if save_report:
                    fm.save_report(str(freq_var[-7:-4]),save_report_path)

    if save_variables :
        with cd(save_variables_path):
            save_file(save_variable_prefix+'bg_ppts.npy',bg_ppts)
            save_file(save_variable_prefix+'peak_ppts.npy',peak_ppts)
            save_file(save_variable_prefix+'errors.npy',errors) 

###############
# Main Script #
###############
# All of the sanity check is to make sure that the loop is going through in the 
# order of ppt number 

freq_path = '/Volumes/SHard/Probes/FreqValuesConds/'
loc_path  = '/Volumes/SHard/Probes/chanLocs/'
save_report_path = '/Volumes/SHard/Probes/freq_fit_report/Off'
save_variables_path = '/Volumes/SHard/Probes/fooof_fit_values'
save_variable_prefix = 'off_'

save_variables = True
save_report    = True

run_fooof(freq_path,loc_path,save_report_path,save_variables_path,save_variable_prefix,save_variables
,save_report,freq_range=[1,30])

#%%
#############
# Edit Here #
#############
   
