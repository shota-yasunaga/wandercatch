%% label_epoch
% Script to plot the number of participants available according to some
% threshold
% !!! You need to run getLabelSequence first. !!!

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
label_dir = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Labels_all';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%%%%%%%%%%%%%%%%%%%
% End of Editting  %
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%

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
label_files = util('getLabelFiles',label_dir);


num_files = length(label_files);

num_labels = [];
for i = 1:num_files
    fid = fopen(label_files{i});    
    fgetl(fid); % Skipping the 
    labels = textscan(fid,'%d %s\n');
    labels = labels{2};
    num_labels(i,1) = sum(strcmp('On',labels));
    num_labels(i,2) = sum(strcmp('MW',labels));
    num_labels(i,3) = sum(strcmp('MB',labels));
end

num_labels(:,4) =num_labels(:,2) + num_labels(:,3); % MW + MB


num_available_ppts = [];
ind = 1;
for trials_thresh = 2:30
    disp(getAboveThresh(num_labels,trials_thresh))
    num_available_ppts(ind,:) = getAboveThresh(num_labels,trials_thresh);
    ind = ind +1;
end

disp(num_available_ppts)

figure;
title('Number of participants available according to the threshold of epochs available per ppt')
titles = {'MW vs MB', 'On vs MW', 'On vs MB','On vs Off'};
for i = 1:4
    subplot(2,2,i)
    scatter(2:30,num_available_ppts(:,i))
    title(titles{i})
    xlabel('Threshold of available epochs')
    ylabel('Number of available ppts')
    ylim([0,20])
end
    

%% 
function [whole_matrix] = getAboveThresh(num_labels,trials_thresh); 
    above_thresh = num_labels > trials_thresh;
    mw_vs_mb = sum(above_thresh(:,2) & above_thresh(:,3));
    on_vs_mw = sum(above_thresh(:,1) & above_thresh(:,2));
    on_vs_mb = sum(above_thresh(:,1) & above_thresh(:,3));
    on_vs_other = sum(above_thresh(:,1) & above_thresh(:,4));
    
    whole_matrix = [mw_vs_mb,on_vs_mw,on_vs_mb,on_vs_other];
end


