% spm2eeglab
% script to loop through spm files to eeglab files
% Created by Shota Yasunaga
% shotayasunaga1996@gmail.com

clear all

%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Edit here (if needed, including Constant)
% Directory to the spm files
% *Do not put anything except the data 
spm_dir = '/Users/tand0009/Data/WanderIM/preproc_eeg/';

% % Look at convert_location_mat2eeglab if you don't have one
% location_file_dir = ...
%     '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/EEG/Matlab/chan_loc.xyz';


% Path to save the eeglab set
saving_path = '/Users/tand0009/Data/WanderIM/eeglab/';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/tand0009/WorkGit/projects/inprogress/wandercatch/';

%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
NUM_CHANNELS = 63;
% sampling Rate
Fs = 500;
%%%%%%%%%%%%%%%%%%%
% End of Editting %
%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
% util
addpath(util_dir)
%eeglab
eeglab

%%
%%%%%%%%%%
% Script %
%%%%%%%%%%



files= util('getBehaviorFiles',spm_dir); % It's not behavior, but it's same .mat file.

cd(spm_dir)


for i = 1:length(files)
    if isempty(findstr(files{i},'nfEEG')) || ~isempty(findstr(files{i},'_nfEEG'))
        continue;
    end
    [~,name,ext] = fileparts(files{i});
    fprintf('Processing %s',name)
    load(files{i})
    D.data.fname = sprintf('%s%c%s%s',spm_dir,'/',name,'.dat');
    data = D.data(1:NUM_CHANNELS,:,:);
    EEG = pop_importdata('dataformat','array','nbchan',NUM_CHANNELS,'data','data',...
        'setname',name,'srate',Fs,'pnts',0,'xmin',0); %,'chanlocs',...
%         location_file_dir);

    % Fix the radius value 
    % Please let me know if you find better way to do this
%     D=spm_eeg_load([spm_dir filesep (files{i})]);
%     myCoord=D.coor2D;
%     [th,r,z] = cart2pol(myCoord(1,:),myCoord(2,:),ones(1,NUM_CHANNELS));
%     for c = 1:NUM_CHANNELS
%         x = myCoord(1,c)';
%         y = myCoord(2,c)';
%         EEG.chanlocs(c).radius = 1; %euclid_dist(x,y)*0.50/0.660; %TODO
%     end

    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',name,'filepath',saving_path); % Save the data
end

    
function dist = euclid_dist(x,y)
    dist = sqrt(x^2+y^2);
end


