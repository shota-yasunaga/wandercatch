# Matlab 

## Procedure (Data conversion)

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


## Feature Plotting
