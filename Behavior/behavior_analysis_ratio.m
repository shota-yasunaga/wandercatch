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

% Directroy path to the helper functions
% (getLabelSequence, split_labels_back)
func_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/Behavior';

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

%%%%%%%%%%%%%%%%%%%%%
% End of Editting   %
%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
NUM_PROBES = 64;
NUM_PPT    = 22;

addpath(func_dir)

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

%%
%%%%%%%%%%%%%%%%%%
% Script to plot %
%%%%%%%%%%%%%%%%%%
% This part calls the functions defined below to actually create plots.
% Comment out unnessesary stuff
%
% In general, 
% Blue:on task,Red:mind wandering,Yellow:mind blanking,Purple:not remember

% f = ppt_whole(all,NUM_PPT) % PPT sum across trials (whole)
% 
% f = ppt_quarter(all,NUM_PPT,NUM_PROBES) %  PPT sum across trials (quarter)
% 
% f = time_whole(on_task,wander,blank,not_remember,NUM_PROBES) % across time




%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions that actually plots %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PPT sum across trials (whole)
function f = ppt_whole(all,NUM_PPT)
    summed_all = sum(all,2);
    on_local= summed_all(1:NUM_PPT,:);
    wander_local= summed_all(NUM_PPT+1:2*NUM_PPT,:);
    blank_local = summed_all(NUM_PPT*2+1:3*NUM_PPT,:);
    not_remember_local = summed_all(NUM_PPT*3+1:end,:);

    f = plot_scat_labels(on_local,wander_local,blank_local,not_remember_local,'total',NUM_PPT);
end

%% PPT sum across trials (quarter)
function f = ppt_quarter(all,NUM_PPT,NUM_PROBES)
    quarter = NUM_PROBES/4;
    for i = 0:3
        summed_all = sum(all(:,(1+quarter*i):quarter*(i+1)),2);
        on_local= summed_all(1:NUM_PPT,:);
        wander_local= summed_all(NUM_PPT+1:2*NUM_PPT,:);
        blank_local = summed_all(NUM_PPT*2+1:3*NUM_PPT,:);
        not_remember_local = summed_all(NUM_PPT*3+1:end,:);
        title_name = sprintf('%d/4',i+1);
        f = plot_scat_labels(on_local,wander_local,blank_local,not_remember_local,title_name,NUM_PPT);
    end
end


%% Num probes over num_ppt
function f = time_whole(on_task,wander,blank,not_remember,NUM_PROBES)
    on_time           = sum(on_task);
    wander_time       = sum(wander);
    blank_time        = sum(blank);
    not_remember_time = sum(not_remember);

    f = figure;
    plot(1:NUM_PROBES,on_time');
    title('over time')
    hold on
    plot(1:NUM_PROBES,wander_time);
    plot(1:NUM_PROBES,blank_time);
    plot(1:NUM_PROBES,not_remember_time);
end


%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions %%
%%%%%%%%%%%%%%%%%%%%%%

function f = plot_scat_labels(on_local,wander_local,blank_local,not_remember_local,title_name, num_ppt)
% getLabelSequence
% Input: 
% folder_path ... path to the behavioral data matlab vairables
% num_trials  ... number of trials per participant
%                 currently only support for the same amount of trials
% 
%
% Output: 
% Matrix that contains
% Row... Participant
% Column... from trial 1 to trial 64
% 

% c = categorical({'On Task', 'MW','MB','Not Remember'});
f = figure;
scatter(linspace(-1.45,-0.55,num_ppt),on_local','filled');
title(title_name)
hold on
scatter(linspace(-0.45,0.45,num_ppt),wander_local','filled');
scatter(linspace(0.55,1.45,num_ppt),blank_local,'filled');
scatter(linspace(1.55,2.45,num_ppt),not_remember_local','filled');

end


