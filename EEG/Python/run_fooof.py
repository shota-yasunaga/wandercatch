from fooof import FOOOF
from fooof import FOOOFGroup
import numpy as np
from mat2python import getFreqValues

testvalues = getFreqValues('/Volumes/SHard/Probes/freqValues/freq_rmchan_rmep_pPR_ffefspm_S301.mat')


########### Trying single ##############
# testvalues = 10**np.array(testvalues[3])
# spectrum = np.array(list(range(0,51)))

# # Initialize FOOOF object
# fm = FOOOF()

# # Define frequency range across which to model the spectrum
# freq_range = [0, 50]
# print(len(spectrum))
# print(len(testvalues))
# # Model the power spectrum with FOOOF, and print out a report
# fm.report(spectrum, testvalues, freq_range)



############## Trying Group #############
# Initialize a FOOOFGroup object, specifying some parameters
# fg = FOOOFGroup(peak_width_limits=[1.0, 8.0], max_n_peaks=8)

# # Fit FOOOF model across the matrix of power spectra
# spectrum = np.array(list(range(0,51)))
# testvalues = 10**np.array(testvalues)

# fg.fit(spectrum, testvalues)


# print('run fit...........')
# print('------------------')
# # Create and save out a report summarizing the results across the group of power spectra
# fg.save_report()

# # Save out FOOOF results for further analysis later
# fg.save(file_name='fooof_group_results', save_results=True)
