%% epoch_w_labels
% Script to create eeglab dataset that is epoched based on the labels 
% (On task, Mind Wandeirng, Mind Blanking)
% You need to have all of the behavior data 
% 
% By Shota Yasunaga
% 
% shotayasunaga1996@gmail.com

% prefixes
% pr... probe
% tr.. trials

clear
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% EEG data
% The script is dependent of the fact taht you put S (capital) in front of
% the number of participants
% if you want to change it, you need to look at around "strtok"
eeg_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/eeglab';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% path to save data
on_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed_conds/On'; % On Task
mb_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed_conds/MB'; % Mind Blanking
mw_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed_conds/MW'; % Mind Wandering
off_dir  = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed_conds/Off'; % Off Task


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

eeg_files    = util('getEEGFiles',eeg_dir);

num_files = length(eeg_files);

names={num_files};
num_on=[];
num_mw=[];
num_mb=[];
num_off=[];
for i = 1:num_files
    eeg_file  = eeg_files(i);
    [~,name,~] = fileparts(char(eeg_file)); % get the name

    % Actually getting labels
%     behave_file = behave_files{contains(behave_files,pat_num)}; %MAybe wrong    
%     labels = util('getProbeLabels',behave_file);
    
    % eeglab
    
    EEG = pop_loadset('filename',eeg_file);  
    
    labels = getLabels(EEG);
    names{i}=name;
    on_inds  = find(strcmp(labels,'On')); num_on(i) = length(on_inds);
    mw_inds  = find(strcmp(labels,'MW')); num_mw(i) = length(mw_inds);
    mb_inds  = find(strcmp(labels,'MB')); num_mb(i) = length(mb_inds);
    off_inds = [mw_inds mb_inds]; num_off(i) = length(off_inds);
    select_with_inds(EEG,on_inds,'ON',name,on_dir);
    select_with_inds(EEG,mw_inds,'WM',name,mw_dir);
    select_with_inds(EEG,mb_inds,'MB',name,mb_dir);
    select_with_inds(EEG,off_inds,'Off',name,off_dir);
end
disp('On')
disp(num_on)
disp('MW')
disp(num_mw)
disp('MB')
disp(num_mb)
disp('Off')
disp(num_off)
disp('Name')
disp(names)



function EEG = select_with_inds(EEG, inds, label,name, saving_path)
    if not(isempty(inds))
        % to choose the trials and save to the right path
        EEG = pop_select(EEG,'trial',inds);
        EEG.setname = label;
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',sprintf('%s%s',label,name),'filepath',saving_path); % Save the data
    end
end

function labels = getLabels(EEG)
    labels={};
    for i = 1:length(EEG.epoch)
        labels{i} = EEG.epoch(i).response;
    end
end