function labelMat = getLabelSequence(folder_path, num_probes)
% getLabelSequence
% Input: 
% folder_path ... path to the behavioral data matlab vairables
% num_trials  ... number of trials per participant
%                 currently only support for the same amount of trials
% 
%
% Output: 
% Matrix that contains
% Row... Participant
% Column... from trial 1 to trial 64
% 

NUM_QUESTION_LABEL=2; %(second question)
NUM_QUESTIONS = 7; % 7 Qs for each probe
LABEL_IND = 9; %

files = dir(folder_path);
files=files(~ismember({files.name},{'.','..'}));

cd(folder_path)

num_ppt = length(files);

labelMat = zeros(num_ppt,num_probes);
for ppt = 1:num_ppt
   load(files(ppt).name)
   labelMat(ppt,:) = all_probe_responses(NUM_QUESTION_LABEL:NUM_QUESTIONS:end,LABEL_IND)';
end