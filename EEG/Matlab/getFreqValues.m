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
eeg_dir      = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes';
folder_names = {'On','Off','MW'};

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% to save eeg file if you wnat to overwrite,same as eeg_dir
saving_dir  = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/freq_plots';

%%
%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
% threshold
% threshold to detect noisy channels
% if channels have the potential more than the threshold, it's a noise


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
for cond = folder_names
    disp(cond{1}) % Sanity Check
    col = col+1;
    current_folder = util('constructPath',eeg_dir,cond{1});
    eeg_files      = util('getEEGFiles',current_folder);
    num_files      = length(eeg_files);
    for i = 1:num_files 
        EEG = pop_loadset('filename',eeg_files(i));
        disp(EEG.filename) % Sanity check
        plot_decomp(EEG,i,col,cond{1}) % Plot frequency decomposition
    end
end

for i=1:num_files 
    f = figure(i);
    set(f,'rend','painters','pos',[0 0 1200 900])
    [~,name,~] = fileparts(eeg_files{i});
    saveas(f, fullfile(saving_dir, char(['freq_' name])), 'png'); % save it
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Main functino to plot %
%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_decomp(EEG,fid, col,label)
    % Plot before probe and after probe on figure(fid)
    % (frequency decomposition between [1 40].
    % col specifies the column
    % label specifies which condition it was (for title)
    figure(fid)
    subplot(2,3,col)
    title([label ' before'])
    pop_spectopo(EEG, 1, [0  10000], 'EEG' , 'freq', [10 22 30], 'freqrange',[1 40],'electrodes','off');
    subplot(2,3,3+col)
    title([label ' after'])
    pop_spectopo(EEG, 1, [10000  20000], 'EEG' , 'freq', [10 22 30], 'freqrange',[1 40],'electrodes','off');
end


function [eegspecdB,freqs,compeegspecdB,resvar,specstd] = get_decomp_values(EEG)
    
    [eegspecdB,freqs,compeegspecdB,resvar,specstd] = spectopo(EEG.data,5001,
end


