% TODO: Wait... I made it so that it can run on each trials, but it's same
% for all of it...... I'm stupid....

clear 
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the 
% *Do not put anything except the data under the directory


% EEG data
% directory that contains eeglab dataset
% This script loops through the folder under eeg_dir in folder_names
one_folder = false;
 
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes';
folder_names = {'On','Off','MW'};

saving_var   = '/Volumes/SHard/Probes/chanLocs';

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


%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
if one_folder
    run_on_folder(eeg_dir,saving_var)
else
    for cond = folder_names
        disp(cond{1})
        current_folder = util('constructPath',eeg_dir,cond{1});
        current_saving_folder = util('constructPath', saving_var,cond{1});
        run_on_folder(current_folder,current_saving_folder)
    end
end

function run_on_folder(eeg_folder,saving_var)
    eeg_files = util('getEEGFiles',eeg_folder);
    num_files = length(eeg_files);
    % Same TODO as getFreqValues
    
    for i = 1:num_files
        EEG = pop_loadset('filename',eeg_files{i});
        [~,name,~] = fileparts(eeg_files{i});
        labels = chanLabels(EEG);
        saving_local = util('constructPath',saving_var,['chans_', name]);
        save(saving_local,'labels')
    end
end

function labels = chanLabels(EEG)
    labels = {length(EEG.chanlocs)};
    for i = 1:length(EEG.chanlocs)
        labels{i}=EEG.chanlocs(i).labels;
    end
end 


