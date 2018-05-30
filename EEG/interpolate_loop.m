%% interpolate_loop
% Script to detect noisy channels and interpolate
% 
% By Shota Yasunaga
% 
% shotayasunaga1996@gmail.com

clear
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the 
% *Do not put anything except the data under the directory


% EEG data
% directory that contains eeglab dataset
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
% threshold
% threshold to detect noisy channels
% if channels have the potential more than the threshold, it's a noise
THRESHOLD = 180


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



%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
% eeglab %import all of the necessary functions %TODO uncomment out
eegfiles = util('getEEGFiles',eeg_dir);

func_wrapper = @(filename) detect_noisy_channel(char(filename),THRESHOLD);

% chaninds = arrayfun(func_wrapper,eegfiles, 'UniformOutput', false)

chan_inds = detect_noisy_epoch(char(eegfiles(1)),THRESHOLD);


%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function to call %
%%%%%%%%%%%%%%%%%%%%%%%%%


function chan_inds = detect_noisy_channel(filename,threshold)
    
    EEG = pop_loadset('filename',filename);
    noise_flags = EEG.data>threshold;
    chan_noise_flags = sum(sum(noise_flags,2),3)>10; %TODO: what should I do?
    chan_inds = find(chan_noise_flags)
end



function chan_inds = detect_noisy_epoch(filename,threshold)
    EEG = pop_loadset('filename',filename);
    noise_flags = EEG.data>threshold;
    chan_noise_flags = sum(noise_flags,2);
    chan_noise_flags = chan_noise_flags>1
    
    chan_inds = sum(chan_noise_flags,1)
end

