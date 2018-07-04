# wandercatch
Wanderlust project (analyses functions)

## Meta Information
Main Editor: Shota Yasunaga

Email      : shotayasunaga1996@gmail.com

Institution: Monash University  (research conducted)/Pitzer College     (home institute of Shota)

Last Modified [2018-06-04 12:58]  


## Behavior
Functions for behavioral analysis

Information about the data index is at the bottom
- behavior_overview
  - What it does

  Plot what kind of events there were during 20 seconds before the probes. The trials go from left to right, top to bottom.
  (The figure generated contains 64 probes and it generates this for each participant)
  - Legend
    - The plot Background 
        - White  ... On Task
        - Yellow ... Mind Wandering
        - Blue   ... Mind Blanking
        - Grey   ... Didn't remember
    - The events 
        - Green ... Trial End
        - Red   ... Questions
        - Blue  ... Trial Start
        - Black ... Others

- transition_analysis
    
    Plots how the state (On task, Mind Wandering etc) transitioned. 
    - getLabelSequence.m
    Helper function for the transition_analysis. It returns the labels of probes
- behavior_analysis_ratio

    Plots information related how much of probes were "on task", "mind wandering", etc. 

- behavior_response_time.m

  Under development... might not finish...

- plot_conds_num_available
  
  plot number of participants available based on threshold for number of conditions for each conditions. 
  It plots 
  1. Number of available participants for cond vs cond (both conds have to have above threshold epochs available)
  2. Numbr of available participants for each cond


## EEG
Functions to conduct eeg data analysis

### Procedure
0. convert_location_mat2eeglab.m create location file (skip if you have .xyz file. in this repository)
1. spm2eeg.m ... convert spm to eeglab structure
2. label2txt.m ... get the labels of epochs that are available for eeglab
3. label_epoch.m ... labels epochs of eeglab dataset
4. interpolate_loop.m ... clean the data and average reference
5. epoch_w_labels.m ... create new datasets based on the epochs
6. plot_freq_loop.m ... create frequency decompositions maps
6. getFreqValues.m
7. fit_fooof_script.py (from here,it's python)
8. freq_power_comparison.py,plot_fooof_vs_labels.py,peak_power_cmp.py



### Functions
### Matlab
- util.m
  Gathered common operations for scripts

  Syntax is out = util('FunctionName', inputs)
  - getBehaviorFiles(dir)... get the .mat files under dir
  - getEEGFiles (dir) ... get the .set files under dir
  - getProbeLabels(file)... get the labels(On,MW, etc) based on behavior file
  - getLabelFiles(dir)... get .txt files under dir
  - constructPath(parent, child) ... construct path that is parent/child

- AMICA_Wrapper_files

  Run AMICA for each files in a directory
- chan_loc.xyz

  Channel location file for eeglab
- convert_location_mat2eeglab

  You don't have to run this. This creates chan_loc.xyz(above)
- epoch_w_labels

  Script to create eeglab dataset that is epoched based on the labels 
  (On task, Mind Wandeirng, Mind Blanking) and save them to folders accordingly.

  ! Each Epochs have to be labels beforehand. 
- getFreqValues
  
  Scripts to save the values of the power spectrum
  
  DEVELOPPING...
- interpolate_loop.m

  1. Remove bad epochs --> Remove bad channels, both using voltage threshold. For bad epochs, I also have threshold for number of bad channels. If an epoch has more than threshold bas channels, the script reject that epoch.
  2. interpolate removed channels --> average reference --> remove the channels again

- label_epoch.m
  
  Script to label epochs based on the labels that are made with label2txt

- label2txt.m
  
  Based on behavior data, construct text file that is available to eeglab to label epochs

- plot_freq_loop.m
  
   loop to plot frequency decomposition 
 Create a plot/ppt that contains 
 Before Proebe Cond1 Cond2 Cond3 
 After Probe   Cond1 Cond2 Cond3
 (Create 6 plots in 1 figure)

- spm2eeglab.m

  Import spm data to eeglab
  ! It only converts the data and channel location (including epochs but not the labels of epochs)
  ! You need to have .xyz file in order to do the loation convertion

- Features Dimension: (conds,channels,freqneucy bins,time bins)



#### Python
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

##### Helper Functions

- fooof_vars_methods.py

  Methods useful for extracting values from the fooof fitted values(finding alpha, finding amplitude of alpha

- util.py
  
  have different useful functions for mostly classifiers. It also has cd class that deals with moving directories

- mat2python.py

  helper methos to read matlab variables and convert it to be able to use for the python script. It has usefult iterators, too. 

- plot_methods.py
  
  helper methods for plooting scatter plots


## Classifiers

TODO: I need to restructure this because the function names are not appropriate

- SVM.py
  This has all of the classes that I'm using to verify the classifiers.

- SVM_all_pts_together.py
  
  run SVM with all of the participants' features together. 

- SVM_ppts_loop.py

  run classifications on different participants in a loop

- find_coefficients.py

  Run SVM with linear kernel and see the coefficients 
  


#### Behavioral Data

- all_task_responses
 1. num_blocks
 2. num_trials
 3. doesn't_matter
 4. block_type ... 1: verbal task 2: spatial
 5. direction... 1 is right (forward),2 is lest(backward)
 6-9 ... right answer

 10-12... keyboard num (response)

 14-17... corrected response

 18-21... correctness-> NaN means there was probe
    --> no response time


- all_task_times
 1. Block Num
 2. Trial Num
 3. ??
 4. type (spatial/word)
 5. all 1s
 6. duplicate of 4
 7. ??
 8. ??
 9. Onset Trial
 10. Fixation
 11. Onset Stimulus presentation
 12. end of trial
 13. Onset que (arrow)
 14. Second arrow
 15. Num response
 - \*verbal  --> question onset is for each word
 16. Question for the first arrow
 17. Question for the second arrow
 18. Response for the first arrow
 19. Response for the second arrow
- \*spatial --> question onset is only one timing
 16. NaN
 17. NaN
 18. Answer for the first arrow
 19. Answer for the second arrow

- all_probe_response
 There are 7 questions for each probe
 1. Probe Num
 2. Block Num
 3. Trials Num (within a block)
 4. Probe Num (duplicate of 1)
 5. Question Num 
 6. Row keyboard Number
 7. Onset of Question
 8. Onset of the reaction
 9. What the answer was (for the question)
 
 For 2nd question (Labels)
 (1) Off-Task (2) Blank (3) Don''t Remember (4)


- all_probe_times
 1. Probe Num
 2. Block Num
 3. In which trial, this probe happened (WITHIN BLOCKS)
 4. Probe Num within the block
 5. Onset Probe
 6. Onset Response
