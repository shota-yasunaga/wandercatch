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

files = dir(file_dir);
files=files(~ismember({files.name},{'.','..'}));

cd(file_dir)
