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
label_dir = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/Labels';

% EEG data
% The script is dependent of the fact taht you put S (capital) in front of
% the number of participants
% if you want to change it, you need to look at around "strtok"
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%%%%%%%%%%%%%%%%%%%
% End of Editting  %
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
TRIALS_THRESH = 10;

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
%%
above_thresh = num_labels > TRIALS_THRESH;

mw_vs_mb = sum(above_thresh(:,2) & above_thresh(:,3));
on_vs_mw = sum(above_thresh(:,1) & above_thresh(:,2));
on_vs_mb = sum(above_thresh(:,1) & above_thresh(:,3));
on_vs_other = sum(above_thresh(:,1) & above_thresh(:,4));


figure;
bar([mw_vs_mb,on_vs_mw,on_vs_mb,on_vs_other])
set(gca,'xticklabel',{'MW-MB', 'On-MW','On-MB','On-Off'});

figure;
bar(sum(num_labels,1))
set(gca,'xticklabel',{'On', 'MW','MB','Off'});




