% behavior_analysis_ration
% analysis script for analyzing information about labels ratio (MW,MB, etc)
% 
% By Shota Yasunaga
% 22/May/18
% shotayasunaga1996@gmail.com
%
% Requirements: 
%   getLabelSequence


%% Preparation
%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% Directory to the files
% *Do not put anything except the data 
file_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Behavior_Data';

% Directroy path to the helper functions
% (getLabelSequence, split_labels_back)
func_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/Behavior';

% Directory where you want to save the figure
% saving_dir = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/Plots';

%%%%%%%%%%%%%%%%%%%%%
% End of Editting   %
%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%
% Constant %
%%%%%%%%%%%%
NUM_PROBES = 64;
NUM_PPT    = 22;

addpath(func_dir)



%%
%%%%%%%%%%%%%%%%%%%$
% Useful Variables %
%%%%%%%%%%%%%%%%%%%$
labelMat = getLabelSequence(file_dir, NUM_PROBES);
% (1)On-Task (2) Mind Wandeing (3) Blank (4) Don''t Remember

% % Plotting the whole 
% transMat = ppt_transition(labelMat);
% f = ppt_transition_plot(transMat);
% 
% % Plotting transition probability
% transMat = transMat./sum(transMat,2);
% transMat(isnan(transMat)) = 0
% f = ppt_transition_plot(transMat);

% Plot the transition arross time
f = ppt_transition_quarter_plot(labelMat);

function f = ppt_transition_plot(transMat)
    NUM_PPT = size(transMat,3);
    plot_length = ceil(sqrt(NUM_PPT+1));
    f = figure();
    for i = 1:NUM_PPT
        subplot(plot_length,plot_length,i)
        g = digraph(transMat(:,:,i));
        LWidth = 5*g.Edges.Weight/max(g.Edges.Weight);
        plot(g,'EdgeLabel',g.Edges.Weight,'LineWidth',LWidth, 'Layout', ...
        'circle','MarkerSize',10,'NodeColor','red', 'NodeLabel',{'On','MW','MB','??'})
        title(sprintf('Participant %d',i))
    end
    subplot(plot_length,plot_length,NUM_PPT+1)
    g = digraph(mean(transMat,3));
    LWidth = 5*g.Edges.Weight/max(g.Edges.Weight);
    plot(g,'EdgeLabel',g.Edges.Weight,'LineWidth',LWidth, 'Layout', ...
        'circle','MarkerSize',10,'NodeColor','red', 'NodeLabel',{'On','MW','MB','??'})
    title('Average Across ppts')
end

function f = ppt_transition_quarter_plot(labelMat)
    %CAUTION: has to be divisible by 4
    %TODO: fix it
    QUARTER = size(labelMat,2)/4;
    for i = 1:4
        if i ==4
            label_quarter = labelMat(:,1+QUARTER*(i-1):end);
        else
            label_quarter = labelMat(:,1+QUARTER*(i-1):QUARTER*i+1);
        end
        transMat=ppt_transition(label_quarter);
        f = ppt_transition_plot(transMat);
        set(f,'Name',sprintf('%d/4',i),'Numbertitle','off');
    end
end



%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions %%
%%%%%%%%%%%%%%%%%%%%%%

%% Markov Chain ish
% The transition probability for each ppt
% Assuming there are 4 sates
function transMat = ppt_transition(labelMat)
    NUM_PPT = size(labelMat,1);
    transMat = zeros(4,4,NUM_PPT);
    for ppt = 1:NUM_PPT
        ppt_mat = labelMat(ppt,:); 
        ppt_mat = ppt_mat(not(isnan(ppt_mat))); % removing NaN
        ppt_trans = zeros(4,4); %4 ... states
        NUM_PROBES = length(ppt_mat);
        for i = 1:NUM_PROBES-1
            ppt_trans(ppt_mat(i),ppt_mat(i+1)) = ppt_trans(ppt_mat(i),ppt_mat(i+1)) + 1;
        end
        ppt_trans = ppt_trans/NUM_PROBES;
        transMat(:,:,ppt) = ppt_trans;
    end
    
end