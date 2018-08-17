# wandercatch
This project's aim is to classify "Mind Wandering" and "On Task" based on the neural signal.

## Information
Main Developer: Shota Yasunaga

Supported by: Thomas Andrillon, Nao Tsuchiya

Email      : shotayasunaga1996@gmail.com

Institution: Monash University  (research conducted)/Pitzer College     (home institute of Shota)

## Reqiurements 
List of software/packages you need. Let me know if I was missing something

I have used the versions written here, but other versions might work as well. Who knows.
### Matlab
Matlab 2016 (other versions might or might not work)

eeglab (14.0.0)

fieldtrip 



### Python
Python (3.6.2)

matplotlib (2.2.0)

scipy (1.1.0)

numpy (1.14.3)

sklearn 0.19.1 (For classifiers related work)

[fooof](https://github.com/voytekresearch/fooof) (for the power spectram peak/baseline analysis)

## Folder Structure
  There are READMEs for each folder. So, take a look at that for the detail.

  - /[Behavior](https://github.com/andrillon/wandercatch/tree/master/Behavior) ... Behavior analysis
    - contains Matlab files to deal with behavior analysis. For detail, look at the README there. 
  - /[EEG](https://github.com/andrillon/wandercatch/tree/master/EEG) ... EEG data analysis
    - /[Matlab](https://github.com/andrillon/wandercatch/tree/master/EEG/Matlab)
      - /[DataHandling](https://github.com/andrillon/wandercatch/tree/master/EEG/Matlab/DataHandling) ... converts/organizes data
      - /[FeatureExtraction](https://github.com/andrillon/wandercatch/tree/master/EEG/Matlab/FeatureExtraction)... extracts features
      - /[Plots](https://github.com/andrillon/wandercatch/tree/master/EEG/Matlab/Plots) ... plots EEG-related staff
    - /[Python](https://github.com/andrillon/wandercatch/tree/master/EEG/Python)
      - /[Classifiers](https://github.com/andrillon/wandercatch/tree/master/EEG/Classifiers) ... classification scripts
      - helper files for classifications/data conversion and some analysis including FOOOF algorithm.
    - /[thomas](https://github.com/andrillon/wandercatch/tree/master/EEG/thomas)
      - I don't know :) 
    - history.txt ... incomplete note of my analysis procedure. 

  - README.md ... This README file
  - util.m     ... commom operations for all
  - .gitignore ... ignore files

## Quick Guide
  - Feature Extraction 

    EEG/Matlab and EEG/Python. Look at Precedure

  - Classification 
    
    EEG/Python/Classifiers
  - Behavior Analysis

    Look at /Behavior
  - Frequency Plots with std

    After pre-processing (data conversion), EEG/Python/\*.py

## Procedure

Proceude examples for EEG data analysis. For about behavior data file, look at Behacior Data Sectio below.
You are generally required to do the pre-processing regardless of the cleaning or not. Most of the processes are labeling and converting data. 

### Preprocessing

1. spm2eeg.m ... convert spm to eeglab structure
2. label2txt.m ... get the labels of epochs that are available for eeglab
3. label_epoch.m ... labels epochs of eeglab dataset
4. interpolate_loop.m ... clean the data (rough) and average reference (**only if you want to clean the data**)
5. epoch_w_labels.m ... create new datasets based on the epochs


### Other operations

6. plot_freq_loop.m, getFreqValues.m ... create frequency decompositions maps, get the values of that.
7. fit_fooof_script.py (from here,it's python)
8. freq_power_comparison.py,plot_fooof_vs_labels.py,peak_power_cmp.py

## Behavioral Data

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

