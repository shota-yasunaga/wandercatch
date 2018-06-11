import os
from mat2python import getFreqValuesVec,getChanLocs
import numpy as np
import matplotlib.pyplot as plt

## THIS SCRIPT IS NOT GENERALIZABLE
## THIS ONLY WORKS FOR THE CASE WHEN THERE ARE 15 PPTS

def plot_cond(freq_path,loc_path,color,end_freq=30):
    freq_dir = os.fsencode(freq_path)
    loc_dir  = os.fsencode(loc_path)
    subplot_values = 1
    for files in zip(os.listdir(freq_dir),os.listdir(loc_dir)):
        print(files) # Sanity Check. 
        freq_var = os.path.join(freq_dir, files[0])
        loc_var  = os.path.join(loc_dir, files[1])
        plt.subplot(3,5,subplot_values)
        power,spectrum = getFreqValuesVec(freq_var)
        chanlocs = getChanLocs(loc_var)
        if ('Oz' in chanlocs):
            oz_ind = chanlocs.index('Oz')
            spectrum = list(np.array(spectrum).flatten())
            last_ind = spectrum.index(end_freq) # TODO: escape when there is no such value
            plt.plot(spectrum[0:last_ind], power[oz_ind][0:last_ind],color)
        else:
            print(files[0], 'did not have Oz')
            input('understood?')
        subplot_values +=1



loc_dir  = '/Volumes/SHard/Probes/chanLocs/'
base_dir   = '/Volumes/SHard/Probes/freqValues'
conds_dir  = '/Volumes/SHard/Probes/freqValuesConds'
conditions = ['On', 'Off']
colors     = ['r','b']

num_ppts = 15 #TODO Generalize it

# Plot the baseline(after probe)
plt.figure()
plot_cond(base_dir,loc_dir,'k')

for cond,color in zip(conditions,colors):
    plot_cond(conds_dir+'/'+cond,loc_dir,color)

plt.show()
