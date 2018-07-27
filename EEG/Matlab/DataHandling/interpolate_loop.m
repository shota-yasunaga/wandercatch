%% interpolate_loop
% Script to detect noisy channels and interpolate
% Then, it also re-reference the electrodes to average
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

% to save eeg file if you wnat to overwrite,same as eeg_dir
saving_dir  = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab_prep';

%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
% threshold
% threshold to detect noisy channels
% if channels have the potential more than the threshold, it's a noise
THRESHOLD = 200; % Reject this epoch if it contains a time point when the voltage is abobe this

EP_THRESH = 5; % Reject this epoch if it contains this many bad channels


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

% chaninds = arrayfun(func_wrapper,eegfiles, 'UniformOutput', false)
names = cell(size(eegfiles));
num_epochs_removed   = [];
num_channels_removed = [];
chan_inds = cell(size(eegfiles));
EEG  = pop_loadset('filename',eegfiles(1));
chanlocs = EEG.chanlocs;

%% Removing bad epochs and channels
for i = 1:length(eegfiles)
    % Getting right file/name 
    eeg_file = eegfiles{i};
    [folder,name,ext] = fileparts(eeg_file);
    fprintf('\n\n')
    fprintf('=====================\n')
    fprintf('Processing %s...\n', name)
    names{i} = name;
    
    % EEG processing
    EEG = pop_loadset('filename',eeg_file);
    
    % Epochs
    noisy_epochs = detect_noisy_epoch(EEG, THRESHOLD,EP_THRESH);
    num_epochs_removed(i) = length(noisy_epochs);
    EEG = remove_epochs(EEG,noisy_epochs);
    
    % Channels
    noisy_channels = detect_noisy_channel(EEG,THRESHOLD);
    chan_inds{i} = noisy_channels;
    num_channels_removed(i) = length(noisy_channels);
    EEG = remove_channels(EEG,noisy_channels);
    % Rereferencing
    EEG = interp_avg_reference(EEG,chanlocs);
    EEG = pop_saveset( EEG, 'filename',[name ext],'filepath',saving_dir); % Overwrite %TODO not overwrite?
end

disp(names)
disp(num_epochs_removed)
disp(num_channels_removed)
disp(chan_inds(i))


%% Interpolating and average referencing Removing back the bad channels 

%% 

%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function to call %
%%%%%%%%%%%%%%%%%%%%%%%%%

%% functions to detect noise
function chan_inds = detect_noisy_channel(EEG,threshold)
    % function to detect noisy channels throughout all of the time points
    noise_flags = EEG.data>threshold;
    chan_noise_flags = sum(sum(noise_flags,2),3)>10; %TODO: maybe change here
    chan_inds = find(chan_noise_flags);
end


function epoch_inds = detect_noisy_epoch(EEG,volt_thresh,epoch_thresh)
    noise_flags = EEG.data>volt_thresh;
    chan_noise_flags = sum(noise_flags,2);
    chan_noise_flags = chan_noise_flags>1;
    epoch_inds = find(sum(chan_noise_flags,1)>epoch_thresh);
end


%% Functions to remove noise
function EEG = remove_epochs(EEG,epoch_inds)
    EEG = pop_select( EEG,'notrial',epoch_inds);
    EEG.setname= sprintf('rmep_%s',EEG.setname); 
    EEG = eeg_checkset( EEG );

end

function EEG = remove_channels(EEG,chan_inds)
    % be careful. If you have removed channels before, 
    % it might not be able to correctly remove the electrodes
    EEG = pop_select( EEG,'nochannel',chan_inds);
    EEG.setname= sprintf('rmchan_%s',EEG.setname); 
    EEG = eeg_checkset( EEG );
end
%% Functions to rereference
function EEG = interp_avg_reference(EEG, chanlocs)
    labels = remaining(EEG);
    EEG = pop_interp(EEG,chanlocs,'spherical');
    EEG = pop_reref(EEG,[]);
    EEG = pop_select( EEG,'channel',labels);
    EEG = eeg_checkset(EEG);
end

function labels = remaining(EEG)
    labels = {length(EEG.chanlocs)};
    for i = 1:length(EEG.chanlocs)
        labels{i}=EEG.chanlocs(i).labels;
    end
end 
