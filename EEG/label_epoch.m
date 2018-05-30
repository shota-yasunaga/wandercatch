%% label_epoch
% run label2txt first and then this one
% this script puts the epoch labels to the EEG structure (eeglab
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
label_dir = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/Labels';

% EEG data
% The script is dependent of the fact taht you put S (capital) in front of
% the number of participants
% if you want to change it, you need to look at around "strtok"
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab_prep';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% 

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

% Partial Mode if set true if you want to run the script even when you
% don't have equal number of files for behavior and eeg
partial_mode = false; 

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
label_files = util('getLabelFiles',label_dir);

eeg_files    = util('getEEGFiles',eeg_dir);

if length(label_files) ~= length(eeg_files) && not(partial_mode)
    error('You need to have same number of files to run the script')
end

num_files = length(eeg_files);

eeglab
for i = 1:num_files
    % Just getting the right files 
    eeg_file  = eeg_files{i};
    [folder,name,ext] = fileparts(char(eeg_file));
    [~,pat_num] = strtok(name,'S');
    pat_num = sscanf(pat_num,'S%s'); % End with number please
    pat_num = ['s' pat_num];
    label_file  = label_files{contains(label_files,pat_num)};
    
    % Actually getting labels
    EEG = pop_loadset('filename',eeg_file);
    EEG = pop_importepoch( EEG, label_file, {'epoch' 'response'},'timeunit',1,'headerlines',1);
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename', eeg_file);
end

