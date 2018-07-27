%% Pairwise_phase_clustering_plot
% This script plots the pairwise phase clustering you obtained from
% Pairwise_phase_clustering.m
%
% By Shota Yasunaga
% 
% shotayasunaga1996@gmail.com

close all
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% WARNING: If you want to change the parameters of the frequency analysis,
%          you need to modify within eeg2phase funciton. 
% 
% EEG data
pcv_dir = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Features/AllPhaseCluster';
conds = {'On','MW'};

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% path to save clustering values
save_dir     = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Plot_phase';



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

pcv_files_conds = [];
for cond = conds
    current_folder = util('constructPath',pcv_dir,cond{1});
    pcv_files      = util('getBehaviorFiles',current_folder);
    pcv_files_conds = [pcv_files_conds;pcv_files];
end
disp(size(pcv_files_conds))

file_length = length(pcv_files_conds(1,:));

for i =  1:file_length
    file1 = char(pcv_files_conds(1,i))
    file2 = char(pcv_files_conds(2,i))
    
    pcv1 = load(char(pcv_files_conds(1,i)));
    pcv2 = load(char(pcv_files_conds(2,i)));
    pcv1 = pcv1.pcv;
    pcv2 = pcv2.pcv;
%     freqs  = [6 7.5:0.5:12 15 24 30];
    freqs = 1:0.5:40;
    occipi = [30 62 61 63 29 31 59 26 58 25 57 24];
    others = [9 10 48 14 49 15 13 19 53 20 1 2 33 34 35 36];
    disp(i)
    [~,name,~] = fileparts(char(pcv_files_conds(1,i)));
    f = figure('name',name(1:end-4));
    set(f,'rend','painters','pos',[0 0 2000 1500])
    plot_pcv(pcv1,pcv2,freqs,occipi,others);
    
    pause(10)
    print(f,fullfile(save_dir,['Phase', name, '.png']),'-dpng','-r500')
    close(f)
end


%% Helper functions

function plot_pcv(pcv1, pcv2,freqs, occipi, others)
    % plot pcv values
    % I should probably make it so that the other function saves the
    % frequency bins to the pcv, too. (TODO)
    % WWARING: you need to have a figure open
    height = length(occipi);
    width = length(others);
    [mean_v1,std_v1] = getStats(pcv1);
    [mean_v2,std_v2] = getStats(pcv2);
    for i = 0:height-1
        for j = 1:width
            id = i*width+j;
            subplot(height, width,id)
            e1 = errorbar(freqs, squeeze(mean_v1(:,:,id)),std_v1(:,:,id));
            e1.Color = 'red';
            hold on;
            e2 = errorbar(freqs, squeeze(mean_v2(:,:,id)),std_v2(:,:,id));
            e2.Color = 'blue';
        end
    end
    
end

function [mean_v, std_v] = getStats(pcv)
    mean_v = mean(pcv);
    std_v = std(pcv);
end
