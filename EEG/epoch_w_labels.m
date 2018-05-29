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
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

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

%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
behave_files = util('getBehaviorFiles',behavior_dir);

eeg_files    = util('getEEGFiles',eeg_dir);

if length(behave_files) ~= length(eeg_files) && not(partial_mode)
    error('You need to have same number of files to run the script')
end

num_files = length(behave_files);

for i = 1:num_files
    eeg_file  = eeg_files(i);
    [~,name,~] = fileparts(char(eeg_file));
    [~,pat_num] = strtok(name,'S');
    pat_num = sscanf('S%d',pat_num);
    load(char(behave_files(i)))
    eeg_file = ee
end

function partial_correction

end