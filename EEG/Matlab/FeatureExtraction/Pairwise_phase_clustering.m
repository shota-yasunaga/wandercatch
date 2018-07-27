%% Pairwise_phase_clustering
% This script computes inTRA-trial phase consistency, using fourier
% transform with tapers ( currently) 
%
% By Shota Yasunaga
% 
% shotayasunaga1996@gmail.com

clear all

%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% WARNING: If you want to change the parameters of the frequency analysis,
%          you need to modify within eeg2phase funciton. 
% 
% EEG data
eeg_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/';
conds = {'On','MW'};

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% path to fieldtrip
fieldtrip_path = '/Users/macbookpro/Matlab/Toolboxes/fieldtrip';
% path to save clustering values
save_dir     = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Features/AllPhaseCluster';



%%
%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
%util
addpath(util_dir)

% eeglab (ver14.0.0 used, but other versions might work as well)
eeglab
addpath(fieldtrip_path)
addpath(util('constructPath',fieldtrip_path,'forward'))

addpath(util_dir)


%%
%%%%%%%%%%
% Script %
%%%%%%%%%%

% file_name = 'WMlbpPR_ffefspm_S301.set';
% file_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/MW';
% '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/MW';
file_name= 'ONlbpPR_ffefspm_S301.set';
file_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/On';

for cond = conds
    current_folder = util('constructPath',eeg_dir,cond{1});
    current_saving_folder = util('constructPath', save_dir,cond{1});
    eeg_files      = util('getEEGFiles',current_folder);
    num_files      = length(eeg_files);
    start = 1;
    for i = start:num_files %TODO: Put it back to num_files 
        [~,name,~] = fileparts(eeg_files{i});
        pcv = eeg2phase(eeg_files{i});
        save_dir_local = util('constructPath',current_saving_folder,['pcv_',name]);
        save(save_dir_local, 'pcv')
    end
end    


%% Helper functions

function pcv = eeg2phase(eeg_path)
    % Modify this function's values to modify the specific parameters of
    % the frequency calculations
    try 
        EEG = pop_loadset(eeg_path);
        field_data = eeglab2fieldtrip(EEG,'preprocessing','none');
    catch 
        eeglab
        EEG = pop_loadset(eeg_path);
        field_data = eeglab2fieldtrip(EEG,'preprocessing','none');
    end

    % choose the time window of interest
    cfg.toilim = [0 22];
    field_data = ft_redefinetrial(cfg,field_data);

    cfg           = [];
    cfg.method    = 'mtmconvol';
    cfg.taper     = 'dpss';
    cfg.trials    =  'all';
    cfg.keeptrials= 'yes';
    cfg.output    = 'fourier';
%     cfg.foi       = [6 7.5:0.5:12 15 24 30];
    cfg.foi       = [ 1:0.5:40];
    cfg.toi       = '50%';
    cfg.t_ftimwin = 0.2*ones([1,length(cfg.foi)]);
    % cfg.t_ftimwin = 4 ./ cfg.foi;
    % cfg.channel   = [ 1 2 33 34 35 36 9 10 48 14 49 15 13 19 53 20 30 62 61 63 29 31 59 26 58 25 57 24];
    occi          = [30 62 61 63 29 31 59 26 58 25 57 24];
    cent          = [9 10 48 14 49 15 13 19 53 20];
    front         = [ 1 2 33 34 35 36];
    cfg.channel   = [occi,cent,front];
    cfg.tapsmofrq = 5;
    freq          = ft_freqanalysis(cfg, field_data);

    % Swap the frequency to do inTRA trial phase consistency
    freq.fourierspctrm = permute(freq.fourierspctrm,[4,2,3,1]);
    freq.time = 1:length(field_data.trial);


    cfg = [];
    cfg.method = 'ppc';

    cfg.channelcmb = combinations(occi,[cent, front],field_data.label);

    connectivity = ft_connectivityanalysis(cfg,freq);
    % 
    % figure;
    % cfg = [];`1
    % cfg.parameter = 'ppcspctrm';
    % ft_connectivityplot(cfg,connectivity)

    pcv = connectivity.ppcspctrm;
    pcv = permute(pcv,[3,2,1]);% permute to get the right format (trials, :)
end

function labels = combinations(comb0,comb1,all_labels)
    labels = cell([length(comb0)*length(comb1),2]);
    ind = 1;
    for c0 = comb0
        for c1 = comb1
            labels{ind,1} = all_labels{c0};
            labels{ind,2} = all_labels{c1};
            ind = ind+1;
        end
    end
end