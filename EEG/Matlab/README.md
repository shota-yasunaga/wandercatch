# Matlab 

## How to use codes

Matlab scripts usually have "Edit here" so that it's clear where you need to modify and where you don't need to. 
You usually need to modify which directories the data is. If you are using my code for all of the procedure, you might not have a problem just by modifying "Edit Here". Otherwise, you probably need to modify the codes. I usually organize the codes so that main scripts don't have much code and helper functions do most of the jobs for readability.

For example, in Pairwise_phase_clustering_plot, it says 
**which script you need to run before**

```Matlab
%% Pairwise_phase_clustering_plot
% This script plots the pairwise phase clustering you obtained from
% Pairwise_phase_clustering.m
%
% By Shota Yasunaga
% 
% shotayasunaga1996@gmail.com
```


and you need to modify this following sections according to your folder structure. 

```Matlab

%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% WARNING: If you want to change the parameters of the frequency analysis,
%          you need to modify within eeg2phase funciton. 
% 
% EEG data
pcv_dir = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Features/AllPhaseCluster';
conds = {'On','MW'};

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% path to save clustering values
save_dir     = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Plot_phase';
```
Following section tells you the dependency to make sure you have all of the packages needed, (util_dir is always the [util.m](https://github.com/andrillon/wandercatch/blob/master/util.m))
```Matlab
%%
%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
%util
addpath(util_dir)
```

Then, there is a main script that usually contains some for loops to loop around 
```Matlab
%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
```
If you need to modify the functions so that it matches with your processing, you should most likely to look at this "Helper Functions" or "Functions" section.

```Matlab
%%
%%%%%%%%%%%%%%%%%%%% 
% Helper functions %
%%%%%%%%%%%%%%%%%%%%
```

## Procedure (DataHandling)

This procedure starts with the spm file and creates new datasets seperated with conditinos.

1. spm2eeg.m ... convert spm to eeglab structure
2. label2txt.m ... get the labels of epochs that are available for eeglab
3. label_epoch.m ... labels epochs of eeglab dataset
4. interpolate_loop.m ... clean the data (rough) and average reference (**only if you want to clean the data**)
5. epoch_w_labels.m ... create new datasets based on the epochs


## Feature Extractions 

These codes are mainly made for classification, but you can use it for regular analysis, too. 

All of the scripts support looping over participants. 

- Pairwise_phase_clustering.m ... this scripts calculate the inTRA-trial pairwise phase consistency (how well channels are in sync, measured by the phase lug between pairts of electrodes)

- getChannelLabels

- getFreqValues


- saveEEG_values.m ... save the raw valtage values as features
    - downsample_loop.m ... downsample the original data and save (meant to be used with saveEEG_values, but you can use it for anything...)

## Plots
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

