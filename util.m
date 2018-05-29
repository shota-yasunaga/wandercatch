function files = util(command,input1)
    % hub for util functions. 
    % This is just to make it look nicer (otherwise, it get too messy with 
    % a lot of different files around)
    if strcmp(command,'getBehaviorFiles') 
        files = getBehaviorFiles(input1);
    elseif strcmp(command, 'getEEGFiles')
        files = getEEGFiles(input1);
    else
        error('Function ''%s'' not defined',command)
    end
    
end


%%Ummm they are the same....
function filenames = getBehaviorFiles(behavior_path)
    filenames = getFiles(behavior_path);
    filenames = filenames(contains(filenames,'.mat'));
end

function filenames = getEEGFiles(eeg_path)
    filenames = getFiles(eeg_path);
    filenames = filenames(contains(filenames,'.set'));
end

%%
%%%%%%%%%%%%%%%%%%%
% Helper Function %
%%%%%%%%%%%%%%%%%%%
function filenames = getFiles(path)
    files = dir(path);
    filenames =[];
    for i=1:length(files)
        filenames = [filenames,string(sprintf('%s/%s',files(i).folder,files(i).name))];
    end 
end