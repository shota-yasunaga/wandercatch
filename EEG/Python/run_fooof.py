import os 
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

####################
# Helper functions #
####################

def findAlpha(peaks):
    for peak_params in peaks:
        peak = peak_params[0]
        if 7<peak and peak <13:
            return peak
    return float('nan')

def findAmps(peaks):
    for peak_params in peaks:
        peak = peak_params[0]
        amp  = peak_params[0]
        if 7<peak and peak <13:
            return amp
    return float('nan')


#%% Plot the relationship betwen bg_ppts and peak_ppts with the labels

label_path = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/Labels'

label_dir=os.fsencode(label_path)
# Get the labels 
files = os.listdir(label_dir)

ons = []
mbs = []
mws = []

for file in files:
    fid = open(os.path.join(label_dir,file))
    print(fid) # Sanity Check
    data = np.loadtxt(fid,delimiter=' ',dtype={'names':('epoch','label'),'formats':('i4','S2')},skiprows=1)
    fid.close()
    pr_labels = list(zip(*data))[1]
    f = lambda byte: byte.decode("utf-8")
    pr_labels = list(map(f,pr_labels))
    ons.append(pr_labels.count('On'))
    mws.append(pr_labels.count('MW'))
    mbs.append(pr_labels.count('MB'))

if len(ons) != len(bg_ppts):
    input('Nah')

first  = lambda arr: arr[0]
second = lambda arr: arr[1]
alpha  = lambda arr:  float('nan') if arr.size == 0 else findAlpha(arr)
alpha_amp =lambda arr: float('nan') if arr.size == 0 else findAmps(arr)   

offset = list(map(first,bg_ppts))
# plt.figure()
# plt.suptitle('Offset')
# plot_scatters(offset,[ons,mws,mbs],['On','MW','MB'],[221,222,223],'Background Offset','Number of labels')

slope  = list(map(second,bg_ppts))
# plt.figure()
# plt.suptitle('Slope')
# plot_scatters(slope,[ons,mws,mbs],['On','MW','MB'],[221,222,223],'Background Slope','Number of labels')

# peaks = list(map(alpha,peak_ppts))
# plt.figure()
# plt.suptitle('Alpha Peak')
# plot_scatters(peaks,[ons,mws,mbs],['On','MW','MB'],[221,222,223],'Alpha Peak','Number of labels')

# amps = list(map(alpha_amp,peak_ppts))
# plt.figure()
# plt.suptitle('Amplitude')
# plot_scatters(peaks,[ons,mws,mbs],['On','MW','MB'],[221,222,223],'Alpha Amplitude','Number of labels')

# plot_scatter(offset,slope,'Offset vs Slope',111,'offset','slope')

plt.show()
