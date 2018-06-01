function out = util(command,varargin)
    % hub for util functions. 
    % This is just to make it look nicer (otherwise, it get too messy with 
    % a lot of different files around)
    % TODO: input number sanity check
    if strcmp(command,'getBehaviorFiles') 
        out = getBehaviorFiles(varargin{1});
    elseif strcmp(command, 'getEEGFiles')
        out = getEEGFiles(varargin{1});
    elseif strcmp(command,'getProbeLabels')
        out = getProbeLabels(varargin{1});
    elseif strcmp(command,'getLabelFiles')
        out = getLabelFiles(varargin{1});
    elseif stcmp(command, 'constructPath')
        out = constructPath(varagin(1),varagin(2));
    else
        error('Function ''%s'' not defined',command)
    end
    
end


%% Actual functions called by util
function filenames = getBehaviorFiles(behavior_path)
    filenames = getFiles(behavior_path);
    filenames = filenames(contains(filenames,'.mat'));
end

function filenames = getEEGFiles(eeg_path)
    filenames = getFiles(eeg_path);
    filenames = filenames(contains(filenames,'.set'));
end

function filenames = getLabelFiles(label_path)
    filenames = getFiles(label_path);
    filenames = filenames(contains(filenames,'.txt'));
end

function labels = getProbeLabels(behave_file)
    load(behave_file, 'all_probe_responses')
    labels = all_probe_responses(2:7:end,9);
end

function result_path = constructPath(parent,child)
    % construct path from the parent and child
    % this just takes care of / insertion
    % Caution! This only works with Mac
    if parent(end) = '/' || parent(end) = '\'
        result_path = [parent child];
    else
        result_path = [parent '/' child];
    end
   
end


%%
%%%%%%%%%%%%%%%%%%%
% Helper Function %
%%%%%%%%%%%%%%%%%%%
function filenames = getFiles(path)
    % grabs all of the files from the path
    files = dir(path);
    filenames =cell(size(files));
    for i=1:length(files)
        filenames{i} = sprintf('%s/%s',files(i).folder,files(i).name);
    end 
end