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
saving_dir   = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/Labels';

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


%! is to run the terminal command

cd(saving_dir)

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


function words = getLabelWords(labels)
    words = cell(size(labels));
    words(labels==1) = {'On'};
    words(labels==2) = {'MW'};
    words(labels==3) = {'MB'};
    words(labels==4) = {'NA'};
end


