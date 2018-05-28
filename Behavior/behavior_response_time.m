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

load(files(1).name)

verbal = all_task_times(all_task_times(:,7) == 1, :);
chuncked = chunk_questions(verbal,4,[16 18], [17,19]);

disp(size(chuncked))

chuncked = delete_nan_row(chuncked);

disp(size(chuncked))

mask = get_mask_whole(all_probe_times,all_probe_responses,all_task_times)

for ppt = 1:length(files)
    
end


function verbal_analysis

end

function spatial_analysis
end

% function to get the matrix for only for verbal response time
% row: probes
% col: QAQAQAQA (Q = question, A = anser)

% % % % function verbalMat = get_verbal_mat(all_task_times)
% % % %     verbal_mask = all_task_times(:,7)==1;
% % % %     all_task_times = all_task_times(verbal_mask,:);
% % % %     
% % % %     order_mask = ... % This is just the information where we should look for the values
% % % %     for row = 1:4
% % % %         for col = [16 18]
% % % %             
% % % %         end
% % % %     end
% % % %         
% % % %     
% % % % end
% % % % 
% % % % function spatialMat = get_spatial_mat()
% % % % end


function chunked = chunk_questions(all_task_times, rows, cols1,cols2)
    % function to change the structure of the function
    % currently, the data is quite hard to work with because of... you know
    % this chunks questions/responses to one row with right oarder 
    % poorly explained
    % input
    %   rows ... number of; rows to iterate through with col1s
    %   
    % output

    NUM_ROWS = size(all_task_times,1);
    chunked = zeros(NUM_ROWS/rows,16); % ummmmm TODO: this is not genrailzed
    for i = 1:NUM_ROWS/rows % should be devisible, otherwise, that's weird
        row_num = 1+ rows*(i-1);% failure to declare variable names in a right way
        n = 1;
        for row = 0:rows-1
            chunked(i,n:n+length(cols1)-1) =all_task_times(row_num+row,cols1);
            n = n+length(cols1);
        end
        for row = 0:rows-1
            chunked(i,n:n+length(cols2)-1) = all_task_times(row_num+row,cols2);
            n = n+length(cols2);
        end
    end
    
end

% function to delte the row that contains Nan
function new_matrix = delete_nan_row(matrix)
    valid_inds = sum(isnan(matrix),2) == 0; % Get the mask
    new_matrix = matrix(valid_inds,:);
end

function mask_struct = get_mask_whole(all_probe_times,all_probe_responses,all_task_times)
    % creates masks for each conditions with trial
    % it creates a structure array in the order On, MW,MB,Not Remember
    
    
    %%%%% TODO: INSANE %%%%%%%%%%%%%
    NUM_PROBES = size(all_probe_times,1);
    prev_ind = 1;
    mask_struct = struct(); % declaratio
    mask_struct(4).val = []; % initialization
    for i = 1:NUM_PROBES
        pr_type = all_probe_responses(((i-1)*7+2),9);
        current_ind = get_time_ind(all_probe_times,all_task_times,i) +3 ;
        mask_struct(pr_type).val = [mask_struct(pr_type).val prev_ind:current_ind];
        prev_ind = current_ind + 5; % skipping the trial where probe happened
    end
    
end

function trial_num = get_time_ind (all_probe_times,all_task_times,i)
    % Function to get the number of row based on the ith probe.
    pr_tr    = all_probe_times(i,3) - 1; % in which trial probe happened (WITHIN A BLOCK)
    pr_block = all_probe_times(i,2); %TODO modify 1 to i
    block_inds = find(all_task_times(:,1)==pr_block)';
    trial_num = find(all_task_times(block_inds,2)==pr_tr, 1); 
end
