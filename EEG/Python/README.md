# wandercatch/EEG/Python

This folder includes all of the EEG-related python script/modeules.

For the **Classifiers**, look at [here](https://github.com/andrillon/wandercatch/tree/master/EEG/Python/Classifiers#classifiers).

## Functions (Modules)


### mat2python

mat2python has all of the feature import/modification related functions. (I have a lot of regrets for the nameings of functions...)

This has some interpretability adbantages since the cahnce performance is 50% (You don't have to compare against the different chance performance based on the subjects). 

List of functions 
- getRawValues
- getGeneralValues
- 

If you want to run Classifications on a new features. 

Functions to pass for [subSampler](https://github.com/andrillon/wandercatch/tree/master/EEG/Python/Classifiers#subsampler)
- 
Helper Functions (not meant to be callled outside although you can if you wish)


### fooof_vars_methods.py

  Methods useful for extracting values from the fooof fitted values(finding alpha, finding amplitude of alpha

### util.py
  
  have different useful functions for mostly classifiers. It also has cd class that deals with moving directories

### plot_methods.py
  
  helper methods for plooting scatter plots

## Scripts
- fit_fooof_script.py

  Script to fit the fooof model for all of the power spectrum under a folder.
  You can save the fitted values/report of fitting

- freq_power_comparison.py

  Script to plot the power spectrum of the differnent conditions for each participant

- plot_fooof_vs_labels.py

  scripts to plot fooof fitted values vs the amount of labels (On,OW,MB,Off)

- peak_power_cmp.py

  to compare the peak (including freqneucy tags and alpha) values between labels(On,MW,MB,etc)

- flicker_analysis.py
  
  script to compare the values of the flickers power values