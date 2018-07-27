clear


% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
%util
addpath(util_dir)

parent_directory = '/Volumes/SHard/Tsuchiya_Lab_Data/Trials/Unprocessed/Downsampled/MW'; % put the name of parent directory
save_to_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Trials/Unprocessed/DownSampled/RawValues/MW';
files = dir(parent_directory);
tic
for dir_ind = 1: length(files)
    dir = files(dir_ind);
    if(endsWith(dir.name, '.set') )
        path = dir.folder;
        EEG = pop_loadset('filename',dir.name,'filepath',path);
        data = EEG.data;
        data = permute(data,[3,1,2]);
        [~,name,~] = fileparts(dir.name);
        saving_local = util('constructPath',save_to_path,['raw_', name]);
        save(saving_local,'data')
        toc
    end
end
