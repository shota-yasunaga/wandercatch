from fooof_vars_methods import findAlpha,findAmps
from mat2python import getFreqValuesVec, folderIterator, chanIndIterator
from plot_methods import plot_scatters,plot_scatter
import matplotlib.pyplot as plt
import numpy as np
from util import cd

def getFlickerFreqs(flickers):
    around_fun = lambda flicker: [flicker-0.1,flicker,flicker+0.1]
    around_flickers = map(around_fun,flickers)
    result_list = []
    for flicker in around_flickers:
        result_list = result_list+flicker
    return result_list

def getDiffs(cond0_itr,cond1_itr,chan_inds_itr,freqs,avoid_flickers=False,flickers=[6,7.5,12,15]):
    
    if avoid_flickers: around_flickers = getFlickerFreqs(flickers) 

    diffs = []
    for freq in freqs:
        power0,spectrum = getFreqValuesVec(next(cond0_itr))
        power1,_ = getFreqValuesVec(next(cond1_itr))
        chan_ind  = next(chan_inds_itr)
        spectrum = [round(sp[0],1) for sp in spectrum] # rounding the spectrum values (to avoid the problem with floating points)
        if np.isfinite(freq):
            ind_peak = spectrum.index(freq) #TODO, if it's not roundable values
            if avoid_flickers: # Avoid the flicker frequencies
                if freq in around_flickers:
                    ind_peaks = [ind_peak-2,ind_peak+2]
                    power1_mean =(power1[chan_ind][ind_peaks[0]] + power1[chan_ind][ind_peaks[1]]) / 2.0
                    power0_mean = (power0[chan_ind][ind_peaks[0]]+power0[chan_ind][ind_peaks[1]])/2.0
                    diffs.append(power1_mean- power0_mean )
                else:
                    diffs.append(power1[chan_ind][ind_peak] - power0[chan_ind][ind_peak])    
            else:
                diffs.append(power1[chan_ind][ind_peak] - power0[chan_ind][ind_peak])

    return diffs

#############
# Edit here #
#############

fooof_vars_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/fooof_fit_values'
## off - on
freq_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/FreqValuesConds/'
loc_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/chanLocs/'
conds = ['Off','On']

##########
# Script #
##########

with cd(fooof_vars_path):
    peak_ppts = np.load('basepeak_ppts.npy')


# Functions to apply for list
alpha  = lambda arr:  float('nan') if arr.size == 0 else findAlpha(arr)
peaks = list(map(alpha,peak_ppts))
peaks = [round(peak,1) for peak in peaks]# round it to 0.1 scale 


folder_iter_wrapper = lambda cond: folderIterator(freq_path + '/'+cond)

#TODO: ughhhh, generalize it... cond1 cond2...
# Get folder iterators for conditions
cond0_itr = folder_iter_wrapper(conds[0])
cond1_itr = folder_iter_wrapper(conds[1])
chan_inds_itr = chanIndIterator(loc_dir)


plt.figure()
# The difference of conditions with respect to alpha(cond1 - cond0)
alpha_diffs = getDiffs(cond0_itr,cond1_itr,chan_inds_itr,peaks,avoid_flickers=True)
plot_scatter(list(range(len(alpha_diffs))), alpha_diffs,'alpha_diffs',231,'ppt_num', conds[0]+'-'+conds[1],regression=False)

flickers = [6,7.5,12,15]
flicker_diffs=[]
ind = 0
for flicker in flickers:
    cond0_itr = folder_iter_wrapper(conds[0])
    cond1_itr = folder_iter_wrapper(conds[1])
    chan_inds_itr = chanIndIterator(loc_dir)
    flicker_diffs.append(getDiffs(cond0_itr,cond1_itr,chan_inds_itr,[flicker]*15))

flicker_diffs.append(alpha_diffs)


plot_scatters(list(range(len(flicker_diffs[0]))),flicker_diffs,['6','7.5','12','15'],[232,233,234,235],'ppt_num',conds[0]+'-'+conds[1],regression =False)

plt.show()