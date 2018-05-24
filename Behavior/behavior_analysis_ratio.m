% behavior_analysis_ration
% analysis script for analyzing information about labels ratio (MW,MB, etc)
% 
% By Shota Yasunaga
% 22/May/18
% shotayasunaga1996@gmail.com
%
% Requirements: 
%   getLabelSequence


%% Preparation
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the files
% *Do not put anything except the data 
file_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Behavior_Data';

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

%%%%%%%%%%%%%%%%%%%%%
% End of Editting   %
%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%
% Fixed values %
%%%%%%%%%%%%%%%%
NUM_PROBES = 64;

%% 
%%%%%%%%%%%%%%%%%%%$
% Useful Variables %
%%%%%%%%%%%%%%%%%%%$
labelMat = getLabelSequence(file_dir, NUM_PROBES);
% (1)On-Task (2) Mind Wandeing (3) Blank (4) Don''t Remember


% Individual Labels
on_task = (labelMat == 1);
wander  = (labelMat == 2);
blank   = (labelMat == 3);
not_remember = (labelMat == 4);

all = [on_task;wander;blank;not_remember];

%% PPT sum across trials (whole)
sum(all,2)


%% PPT sum across trials (quarter)






