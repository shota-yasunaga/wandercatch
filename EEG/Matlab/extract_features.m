%% extract_features
% extract frequency decoding for each epoch for each condition

clear 
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% EEG data
% directory that contains eeglab dataset
% This script loops through the folder under eeg_dir in folder_names

eeg_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed';
conds = {'On','MW'};


% For both
% for multiple folders it would look like saving_var/folder_name
saving_var   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_all_conds';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%
%%%%%%%%%%%%
% Constant %`
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
for cond = conds
    disp(cond{1}) % Sanity Check
    
    % Get the three dimensional features.
    current_folder = util('constructPath',eeg_dir,cond{1});
    current_saving_folder = util('constructPath', saving_var,cond{1});
    eeg_files      = util('getEEGFiles',current_folder);
    num_files      = length(eeg_files);
    start = 1;
    for i = start:num_files %TODO: Put it back to num_files 
        loopEpochs(eeg_files{i},current_saving_folder,1,0,10000);            
    end
    start = 1;
end


function [spec_steps,freqVec] = get_decomp_values(EEG,num_steps,start_frame, end_frame)
    figure;
    windows = linspace(start_frame,end_frame,num_steps+1);
    spec_steps = [];
    for step = 1:num_steps
        start_step = windows(step);
        end_step   = windows(step+1);
        [eegspecdB,freqVec] = pop_spectopo(EEG, 1, [start_step  end_step], 'EEG' , 'freqrange',[1],'electrodes','off','winsize',5000);
        spec_steps(:,:,step)=eegspecdB; %(electrodes,frequency,
    end
    close(gcf) % Close the figure
end

function [features,freqVec] = loopEpochs(filename,saving_dir,num_steps,startFrame,endFrame)
    features = [];
    EEG = pop_loadset('filename',filename);
    num_epoch = EEG.trials;
    for ep = 1:num_epoch
        EPOCH = pop_select(EEG,'trial',ep);
        [spec_steps,freqVec] = get_decomp_values(EPOCH,num_steps,startFrame,endFrame);
        features(ep,:,:,:) = spec_steps;
    end
    [~,name,~] = fileparts(filename);
    saving_local = util('constructPath',saving_dir,['freq_', name]);
    save(saving_local,'features','freqVec')
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