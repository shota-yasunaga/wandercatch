%% extract_features
% extract frequency decoding for each epoch for each condition

clear 
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% EEG data
% directory that contains eeglab dataset
% This script loops through the folder under eeg_dir in folder_names

eeg_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian';
conds = {'On','MW'};
    

% For both
% for multiple folders it would look like saving_var/folder_name
saving_var   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Features';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% Dataflag. If components, 0. Electrodes -> 1.
dataflag = 1;
% window_size ... window size of fourier transform in terms of data points
%                 In terms of seconds, it's window_size / Fs
%                 bandwidth = 1/window_size * Fs
%                 
window_size = 10*500;

% Number of steps to average for the fourier transform
% Without overlaps, it should be (endFrame-startFrame)/num_steps
num_steps = 1;
% 
startFrame = 0;
endFrame = 20*1000; 
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
        %                             num_steps,start_frame,end_frame
        loopEpochs(eeg_files{i},current_saving_folder,dataflag,num_steps,startFrame,endFrame,window_size);            
    end
    start = 1;
end

%% Functions

function [spec_steps,freqVec] = get_decomp_values(EEG,dataflag,num_steps,start_frame, end_frame,window_size)
    figure;
    windows = linspace(start_frame,end_frame,num_steps+1);
    spec_steps = [];
    for step = 1:num_steps
        start_step = windows(step);
        end_step   = windows(step+1);
        % Winsize --> windowsize used by fft. --> decides the resolution
        [eegspecdB,freqVec] = pop_spectopo(EEG, dataflag, [start_step  end_step], 'EEG' , 'freqrange',[0],'electrodes','off','winsize',window_size);
        spec_steps(:,:,step)=eegspecdB; %(electrodes,frequency,
    end
    close(gcf) % Close the figure
end

function [features,freqVec] = loopEpochs(filename,saving_dir,dataflag,num_steps,startFrame,endFrame,window_size)
    features = [];
    EEG = pop_loadset('filename',filename);
    num_epoch = EEG.trials;
    for ep = 1:num_epoch
        EPOCH = pop_select(EEG,'trial',ep);
        [spec_steps,freqVec] = get_decomp_values(EPOCH,dataflag,num_steps,startFrame,endFrame,window_size);
        
        disp(size(features))
        disp(size(spec_steps))
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