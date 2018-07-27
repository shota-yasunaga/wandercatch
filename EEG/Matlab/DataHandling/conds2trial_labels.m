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

% SPM folder
spm_dir      = '/Volumes/SHard/Tsuchiya_Lab_Data/Trials/spm';

% Folder to save the epoch information .txt file
saving_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Trials/Labels';


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





%% trials (There is only trials)    

if trials
    spm_files = util('getBehaviorFiles',spm_dir); % Proably I should rename the funciton. This just takes all of the .mat file
    prev_onset = 0;
    for i = 1:length(behave_files)
        
        % Deal with behavior file
        load(behave_files{i})
        labels = getTrialLabels(all_probe_times,all_task_times,behave_files{i});
        
        [~,name,~] = fileparts(behave_files{i});
        
        disp('#################################')
        disp('#          Sanity Check         #')
        fprintf('# %s #\n',name)
        
        % Deal with spm file
        spm_file    = spm_files{i};
        load(spm_file)
        
        [~,name,~] = fileparts(spm_file);
        skipped_inds = find_discont(D);
        
        fprintf('# %s #\n',name)
        disp('##################################')
        
        input('hey')
        
           
        fid = fopen([name '.txt'],'wt'); %Create a text file to write
        fprintf(fid,'Epoch Response\n'); 
        words = getLabelWords(labels);
        ep_num = 1;
        if length(words) ~= length(labels)
            input('yo. There is something wrong')
        end

        for ep = 1:length(labels)
            if ~(any(skipped_inds == ep))
                fprintf(fid,'%d %s\n',ep_num,words{ep});
                ep_num = ep_num + 1;
            end
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


function inds = find_discont(D)
    % D ... spm data
    inds = [];
    label = D.trials(1).label;
    
    start = 2;
    if strcmp(label, 'unknown')
        label = D.trials(2).label;
        start = 3;
    end
    prev_scan = textscan(label,'B%d_TR%d_%s');
    
    for i = start:length(D.trials)
        label = D.trials(i).label;
        scan = textscan(label,'B%d_TR%d_%s');

        % I'm sorry I'm sorry I'm sorry. This if is just trying to find
        if ((scan{1} == prev_scan{1}) && (scan{2}-prev_scan{2} ~= 1))...
                || ((scan{1} ~= prev_scan{1}) && (scan{2} ~= 1))
            disp('Found discontinuity')
            disp(prev_scan)
            disp(scan)
            disp(i)
            inds = [inds, i];
        end
        prev_scan = scan;
    end
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

