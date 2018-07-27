%% getFreqValues
% Get the frequency values (power values) for each ppt, each chan
input('Are you sure you want to delete all of the figures open?')

clear 
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Are you running this for one folder? otherwise, multiple folders(aka, 
% different conditions)
one_folder = false;

% EEG data
% directory that contains eeglab dataset
% This script loops through the folder under eeg_dir in folder_names
if one_folder
    eeg_folder   = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab_prep';
else
    eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes';
    folder_names = {'On','Off','MW'};
end

% For both
% for multiple folders it would look like saving_var/folder_name
saving_var   = '/Volumes/SHard/Probes/freqValues';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
% threshold
% threshold to detect noisy channels
% if channels have the potential more than the threshold, it's a noise


%%%%%%%%%%%%%%%%%%%%
% End of Editting  %
%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
%util
addpath(util_dir)

% eeglab (ver14.0.0 used, but other versions might work as well)
eeglab 
close all


%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
if one_folder
    eeg_files = util('getEEGFiles',eeg_folder);
    num_files = length(eeg_files);
    for i = 1:num_files
        main_get_save(eeg_files{i},1:51,saving_var,10000,20000)
    end
else
    for cond = folder_names
        disp(cond{1}) % Sanity Check
        current_folder = util('constructPath',eeg_dir,cond{1});
        current_saving_folder = util('constructPath', saving_var,cond{1});
        eeg_files      = util('getEEGFiles',current_folder);
        num_files      = length(eeg_files);
        for i = 1:num_files 
            main_get_save(eeg_files{i},1:51, current_saving_folder,0,10000)            
        end
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Main functino to plot %
%%%%%%%%%%%%%%%%%%%%%%%%%



function [eegspecdB,freqVec] = get_decomp_values(EEG,start_frame, end_frame)
    figure;
    [eegspecdB,freqVec] = pop_spectopo(EEG, 1, [start_frame  end_frame], 'EEG' , 'freqrange',[1],'electrodes','off');
    close(gcf) % Close the figure
end


function main_get_save(filename,range,saving_dir,startFrame,endFrame)
    EEG = pop_loadset('filename',filename);
    %TODO: This is not beautiful... fix it (I set the EEG.setname to be the
    %name of conditions and it's causing problem here. 
    [~,name,~] = fileparts(filename);
    [spectra,freqVec]=get_decomp_values(EEG,startFrame,endFrame);
    spectra = spectra(:,range);
    freqVec = freqVec(range);
    saving_local = util('constructPath',saving_dir,['freq_', name]);
    save(saving_local,'spectra','freqVec')
end

