
%% Dependency
%       eeglab
%       AMICA
%       util.m

clear
file_dir = '/Users/macbookpro/Documents/Tsuchiya_Lab_Data/Probes/eeglab'; % put the name of parent directory

save_eeg_dir = '/Volumes/SHard/Probes/ICA';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';


%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
% Util
%util
addpath(util_dir)

% eeglab
eeglab

% Extract only those that are directories.

num_pca = 64; % number of pca
max_iter = 3000; % number of iterations for AMICA;

eegfiles = util('getEEGFiles',file_dir);

for i = 1:length(eegfiles)   
    
    EEG = pop_loadset('filename',eegfiles(i));

    %%%% Run AMICA and save the data %%%% 
    [W,S,mods] = runamica15(EEG.data(:,:), 'pcakeep', num_pca, 'max_iter', max_iter);
    EEG.icaweights = W;
    EEG.icasphere = S(1:size(W,1),:);
    EEG.icawinv = mods.A(:,:,1);
    EEG.mods = mods;
    EEG.icachansind = [];
    
    new_filename = strcat('AMICA_64_',EEG.filename);
    EEG.setname = strcat('AMICA_',EEG.setname);
    EEG = pop_saveset( EEG, 'filename',new_filename,'filepath',save_eeg_dir ); % Save the data

    %%%% Plotting the result %%%%
    pop_topoplot(EEG,0, [1:64] ,'170613_TRIAL7_LP epochs',[8 8] ,0,'electrodes','off'); % plot components map

    saveas(gca, fullfile(save_eeg_dir, 'map'), 'png'); % save it
    close(gcf) % Close the figure
end
