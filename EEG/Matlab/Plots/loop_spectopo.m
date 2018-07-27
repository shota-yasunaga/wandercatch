%% plot_freq_loop
% loop to plot frequency decomposition
% Create a plot/ppt that contains
% Before Proebe Cond1 Cond2 Cond3
% After Probe   Cond1 Cond2 Cond3
% (Create 6 plots in 1 figure)
% Then, it saves the file to saving_dir/cond(variable defined in
% folder_names)

input('Are you sure you want to delete all of the figures open?')

clear 
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the 
% *Do not put anything except the data under the directory


% EEG data
% directory that contains eeglab dataset
% This script loops through the folder under eeg_dir in folder_names
eeg_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/eeglab_prep';
folder_names = {'/'};

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% to save eeg file if you wnat to overwrite,same as eeg_dir
saving_dir  = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Flickers_plot';

%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
% threshold
% threshold to detect noisy channels
% if channels have the potential more than the threshold, it's a noise'


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
close all


%%
%%%%%%%%%%
% Script %
%%%%%%%%%%
col = 0;

eeg_files      = util('getEEGFiles',eeg_dir);

[even_list,odd_list]=getIndswType(eeg_files)


for i = even_list
    plot_spectopo_save(eeg_files{i},saving_dir,[6,12,15,30])
end

for i = odd_list
    plot_spectopo_save(eeg_files{i},saving_dir,[7.5,12,15,24])
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Main functino to plot %
%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%
% Helper functions %
%%%%%%%%%%%%%%%%%%%%

function plot_spectopo_save(eeg_file,saving_dir,flickers)
    disp(eeg_file)
    EEG = pop_loadset('filename', eeg_file);
    figure;
    pop_spectopo(EEG, 1, [0  10000], 'EEG' , 'freq', flickers, 'freqrange',[1 40],'electrodes','off','freqfac',10)
    [~,name,~] = fileparts(eeg_file);
    saveas(gcf, fullfile(saving_dir, char(['freq_' name])), 'png'); % save it
end

function [even_list,odd_list] = getIndswType(eeg_files)
    num_files = length(eeg_files);
    even_list = [];
    odd_list  = [];
    for i = 1:num_files
        filename = eeg_files{i};
        [~,name,~] = fileparts(filename);
        disp(name)
        ppt_num = str2double(name(end-2:end));
        disp(ppt_num)
        if mod(ppt_num,2) == 0
            even_list = [even_list,i]; 
        else
            odd_list = [odd_list,i];
        end
    end
end

