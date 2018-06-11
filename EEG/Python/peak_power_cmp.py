from fooof_vars_methods import findAlpha,findAmps
from mat2python import getFreqValuesVec, folderIterator, chanIndIterator
import numpy as np
from util import cd

def getDiffs(cond0_itr,cond1_itr,chan_inds_itr,freqs):
    diffs = []
    for freq in freqs:
        power0,spectrum = getFreqValuesVec(next(cond0_itr))
        power1,_ = getFreqValuesVec(next(cond1_itr))
        chan_ind  = next(chan_inds_itr)
        spectrum = [round(sp[0],1) for sp in spectrum]
        if np.isfinite(freq):
            ind_peak = spectrum.index(freq) #TODO, if it's not roundable values
            diffs.append(power1[chan_ind][ind_peak] - power0[chan_ind][ind_peak])

    return diffs


fooof_vars_path = '/Volumes/SHard/Probes/fooof_fit_values'
with cd(fooof_vars_path):
    peak_ppts = np.load('basepeak_ppts.npy')


print(peak_ppts)
input('yo')

alpha  = lambda arr:  float('nan') if arr.size == 0 else findAlpha(arr)

peaks = list(map(alpha,peak_ppts))
peaks = [round(peak,1) for peak in peaks]# round it to 0.1 scale 




## off - on
freq_path = '/Volumes/SHard/Probes/FreqValuesConds/'
loc_dir   = '/Volumes/SHard/Probes/chanLocs/'
conds = ['Off','On']

folder_iter_wrapper = lambda cond: folderIterator(freq_path + '/'+cond)

#TODO: ughhhh, generalize it... cond1 cond2...
cond0_itr = folder_iter_wrapper(conds[0])
cond1_itr = folder_iter_wrapper(conds[1])
chan_inds_itr = chanIndIterator(loc_dir)



print(getDiffs(cond0_itr,cond1_itr,chan_inds_itr,peaks))

flickers = [6,7.5,12,15]
for flicker in flickers:
    cond0_itr = folder_iter_wrapper(conds[0])
    cond1_itr = folder_iter_wrapper(conds[1])
    chan_inds_itr = chanIndIterator(loc_dir)
    print(getDiffs(cond0_itr,cond1_itr,chan_inds_itr,[flicker]*15))
