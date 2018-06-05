%% label2txt
% script to label the epoch of eeglab structure

% Create text file


% read text file and put the information about the epochs

clear
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% *Do not put anything except the data under the directory

% behavior
behavior_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Behavior_Data';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% Folder to save the epoch information .txt file
saving_dir   = '/Volumes/SHard/Trials/Labels';


% Is this about probe or trials?
probe = false;
trials = true;
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
num_files = length(behave_files);
cd(saving_dir)

%% Probe

if probe
    for i = 1:num_files
        % Just getting the right files 
        behave_file = behave_files{i};
        [~,name,~]  = fileparts(behave_file);
        fid = fopen([name '.txt'],'wt'); %Create a text file to write

        labels = util('getProbeLabels',behave_file);

        fprintf(fid,'Epoch Response\n');
        words = getLabelWords(labels);
        for ep = 1:length(labels)
            fprintf(fid,'%d %s\n',ep,words{ep});
        end
    end
end

    

if trials
    prev_onset = 0;
    for i = 1:num_files
        load(behave_files{i})
        labels = getTrialLabels(all_probe_times,all_task_times,behave_files{i});
        
        [~,name,~] = fileparts(behave_files{i});
        
        fid = fopen([name '.txt'],'wt'); %Create a text file to write
        fprintf(fid,'Epoch Response\n'); 
        words = getLabelWords(labels);
        for ep = 1:length(labels)
            fprintf(fid,'%d %s\n',ep,words{ep});
        end
        fclose(fid);
    end
end

function words = getLabelWords(labels)
        words = cell(size(labels));
        words(labels==1) = {'On'};
        words(labels==2) = {'MW'};
        words(labels==3) = {'MB'};
        words(labels==4) = {'NA'};
end

function labels = getTrialLabels(all_probe_times,all_task_times,behave_file)
    num_trials = length(all_task_times)/4;
    labels = [];
    trial_onsets = all_task_times(1:4:end,9);
    
    pr_labels = util('getProbeLabels',behave_file);
    prev_onset = 0;
    for i = 1:64   
        %% Get information about the probe
        pr_onset = all_probe_times(i,5);
        section = trial_onsets < pr_onset & trial_onsets > prev_onset;
        labels(section) = pr_labels(i);
        prev_onset = pr_onset;
    end
end

