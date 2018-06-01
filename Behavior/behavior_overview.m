% Script to plot the relationshipt between trial time pins
% and probe answers (epoched with probes) 
% 
% By Shota Yasunaga
% 22/May/18
% shotayasunaga1996@gmail.com

% prefixes
% pr... probe
% tr.. trials

%% Write function to plot one probe and trial information

%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the files
% *Do not put anything except the data 
file_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Behavior_Data';

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

%%%%%%%%%%%%%%%%%%%%
% End of Editting  %
%%%%%%%%%%%%%%%%%%%%
% This line cleans up the folder a bit
!rm .DS_Store
files = dir(file_dir);
files=files(~ismember({files.name},{'.','..'}));

cd(file_dir)

for ppt = 1:length(files)
    load(files(ppt).name)
    f = figure('name',files(ppt).name(1:end-4));
    title(files(ppt).name)
    for i = 1:64 
        subplot(8,8,i)

        % just for one probe

        %% Get information about the probe
        pr_tr    = all_probe_times(i,3); % in which trial probe happened (WITHIN A BLOCK)
        pr_block = all_probe_times(i,2);
        pr_resp  = all_probe_times(i,6);
        pr_ans   = all_probe_responses(((i-1)*7+2),9);
        pr_onset = all_probe_times(i,5);

        %% Get informtion about the trials
        block_inds = find(all_task_times(:,1)==pr_block)';
        trial_num = find(all_task_times(block_inds,2)==pr_tr, 1);  %Within blocks inds
        trial_num = block_inds(trial_num);%the first column of the all_task_times with respect the probe
        
        %% Helper so that it makes sure that we can make sure we have 20 seconds

        trial_type = all_task_times(trial_num, 4);

        %% Construct event list
        %TODO: need to take care when that if statemenet is true (not enough events
        % available)
        % Bad code... Copy and paste
        event_list = getEventList(all_task_times,trial_type,trial_num);
        time_diff = 0;
        while time_diff <40 && all_task_times(trial_num,1)==pr_block% Just for convinience (to make it run faster) 40 is arbitrary
            trial_num = trial_num -4;
            time_diff = pr_onset - all_task_times(trial_num,9) ;
            event_list = [getEventList(all_task_times,trial_type,trial_num) event_list];
        end
        
        %TODO: Delete this
        debugger = event_list;
        debugger_pr = pr_onset;

        pr_onset = pr_onset - event_list(1);
        event_list = event_list - event_list(1);
        if sum(event_list <0) >0
            disp(event_list)
            input('whyyy')            
        end
        % Get the index of all of the trials that are before the probe
        valid_list = (event_list < pr_onset) & (event_list > (pr_onset - 20));
        events_to_plot = event_list(valid_list);
        if (not(isempty(events_to_plot)) & (length(events_to_plot) ~= 1))
            h = plotStemEvents(events_to_plot,event_list,valid_list,trial_type);
        end
        if (pr_ans == 2) % Mind Wandering
            set(gca,'Color','yellow') 
        elseif (pr_ans == 3) % Mind Blanking
            set(gca, 'Color', 'blue')
        elseif (pr_ans == 4) % Don't remember
            set(gca, 'Color', [0.5 0.5 0.5])
        end
        
    end
    % saveas(gca, fullfile(saving_dir, files(ppt).name(1:end-4), 'png'); % save it
    
    % %% Legend
    % event_lgd = legend(f,{'Arrow','Trial Start', 'Trial End','Others'});
    % title(event_lgd,'Event');
    % chleg = get(hleg,'children');
    % set(chleg(1),'color','r')
    % set(chleg(2),'color','g')
    % set(chleg(3),'color','y')
    % set(chleg(4),'color','r')
    % set(chleg(5),'color','m')
    % set(chleg(6),'color','c')
    % 
    % 
    % 
    % label_lgd = legend(f,{'On Task','MW', 'MB', 'Not Remember'});
    % title(label_lgd,'Labels')
    % chleg = get(hleg,'children');
    % set(chleg(1),'color','r')
    % set(chleg(2),'color','g')
    % set(chleg(3),'color','y')
    % set(chleg(4),'color','r')
    % set(chleg(5),'color','m')
    % set(chleg(6),'color','c')
end


%% 
%%%%%%%%%%%%%%%%%%
% Main Functions %
%%%%%%%%%%%%%%%%%%
function event_list = getEventList(all_task_times,type, trial_num)
    % Function to grab
    if type == 1
         % verbal task 
        % event_list
        %    1      2         3                 4                  13
        % Start, Fixation, Onsert Stimuli, 1st Arrow, Q&As (8), 2nd Arrow,
        % Q&As(8), End 
        %
        event_list = ones(1,22);
        event_list(1:3) = all_task_times(trial_num,9:11); % Start, Onset, Fixation
        event_list(4)   = all_task_times(trial_num, 13); % first arrow
        event_list(5:12) = reshape(all_task_times(trial_num:trial_num+3, [16 18])',[1, 8]);% Q&A
        event_list(13) = all_task_times(trial_num, 18); % second arrow
        event_list(14:21) = reshape(all_task_times(trial_num:trial_num+3, [17 19])', [1,8]);% Q&A
        event_list(22) = all_task_times(trial_num, 12); 
    else % spatial task
        event_list       = ones(1,14);
        event_list(1:3)  = all_task_times(trial_num,9:11); % Start, Onset, Fixation
        event_list(4)    = all_task_times(trial_num, 13); % first arrow
        event_list(5:8)  = all_task_times(trial_num:trial_num+3, 18)'; % As
        event_list(9)    = all_task_times(trial_num, 18); % second arrow
        event_list(10:13)= all_task_times(trial_num:trial_num+3,19)'; %As
        event_list(14)    = all_task_times(trial_num, 12); % End of the trial
    end 
end

%TODOTODOTODOTODOTODOTODOTODOTODOTODO do it for spatial
function h = plotStemEvents(events_to_plot,event_list,valid_list,type)
    events_to_plot = events_to_plot - events_to_plot(1);
    
    if type == 1
        event_num = 22;
        arrow2 = 13;
    else
        event_num = 14;
        arrow2 = 9;
    end

    % mask --- this is just ot help plotting ----
    % (for coloring) %TODO. I need to do this for spatial, too. 
    event_length = length(event_list);
    mods = mod(1:event_length,event_num);
    trial_start = mods==1;
    trial_end   = mods==0;
    arrows      = mods==4 | mods==arrow2;
    others      = ones(1,event_length)-trial_start-trial_end-arrows;


    trial_start  = trial_start(valid_list);
    trial_end   = trial_end(valid_list);
    arrows      = arrows(valid_list);
    others      = others(valid_list);


    sudo_values = [others;trial_start;trial_end;arrows;]';
    sudo_values(sudo_values==0) = NaN;
    % -------

    %% define event list tag

    h = stem(events_to_plot, sudo_values,'filled', 'LineWidth', 2);
    xlim([0 20])
    h(1).Color = 'black';
    h(3).Color = 'green';
    h(2).Color = 'blue';
    h(4).Color = 'red';
end






