%% epoch_w_labels
% Script to create eeglab dataset that is epoched based on the labels 
% (On task, Mind Wandeirng, Mind Blanking)
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
% Directory to the 
% *Do not put anything except the data under the directory

% behavior
behavior_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Behavior_Data';

% EEG data
% The script is dependent of the fact taht you put S (capital) in front of
% the number of participants
% if you want to change it, you need to look at around "strtok"
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% path to save data
on_dir   = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/On'; % On Task
mb_dir   = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/MB'; % Mind Blanking
mw_dir   = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/MW'; % Mind Wandering

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

% Partial Mode if set true if you want to run the script even when you
% don't have equal number of files for behavior and eeg
partial_mode = true;

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
behave_files = util('getBehaviorFiles',behavior_dir);

eeg_files    = util('getEEGFiles',eeg_dir);

if length(behave_files) ~= length(eeg_files) && not(partial_mode)
    error('You need to have same number of files to run the script')
end

num_files = length(eeg_files);

eeglab

names={num_files}
num_on=[];
num_mw=[];
num_mb=[];
for i = 1:num_files
    % Just getting the right files 
    eeg_file  = eeg_files(i);
    [folder,name,ext] = fileparts(char(eeg_file));
    [~,pat_num] = strtok(name,'S');
    pat_num = sscanf(pat_num,'S%s'); % End with number please
    pat_num = ['s' pat_num];
    
    % Actually getting labels
%     behave_file = behave_files{contains(behave_files,pat_num)}; %MAybe wrong    
%     labels = util('getProbeLabels',behave_file);
    
    % eeglab
    
    EEG = pop_loadset('filename',[name ext],'filepath',folder);  
    
    labels = getLabels(EEG);
    names{i}=name;
    on_inds = find(strcmp(labels,'On')); num_on(i) = length(on_inds);
    mw_inds = find(strcmp(labels,'MW')); num_mw(i) = length(mw_inds);
    mb_inds = find(strcmp(labels,'MB')); num_mb(i) = length(mb_inds);
    select_with_inds(EEG,on_inds,'ON',name,on_dir);
    select_with_inds(EEG,mw_inds,'WM',name,mw_dir);
    select_with_inds(EEG,mb_inds,'MB',name,mb_dir);
end
disp(num_on)
disp(num_mw)
disp(num_mb)
disp(names)

function partial_correction

end

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