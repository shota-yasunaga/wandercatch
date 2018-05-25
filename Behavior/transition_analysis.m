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

% Transition Matrix
transMat = ppt_transition(labelMat);

% This just makes it possible to pass this function to function
func_wrapper = @(transMat) calc_trans_prob(transMat); 
%%
%%%%%%%%%%%%
% Plotting %
%%%%%%%%%%%%
% Plotting the whole transition probability
% f = ppt_transition_plot(transMat);


% Plotting transition probability
% calc_trans_prob is coming from the helper function section

% f = ppt_transition_plot(labelMat,func_wrapper);

% Just the avarage version of the above
avgTransMat = ppt_avg_transition(labelMat);
f= transition_plot(func_wrapper(avgTransMat),'average');


%%
%%%%%%%%%%%%%%%%%%%%%%
% Plotting Functions %
%%%%%%%%%%%%%%%%%%%%%%
function f = ppt_transition_plot(labelMat, trans_func)
    % Plot the transition arross time
    % f = ppt_transition_quarter_plot(labelMat);
    % I'm sorry, I overcomplicated a little 
    % trans_func should be function handle that convert transMat to appropriate
    % shape you want
    transMat = ppt_transition(labelMat);
    transMat = trans_func(transMat);
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
    % Plotting average across participants
    avgTransMat = ppt_avg_transition(labelMat);
    avgTransMat = trans_func(avgTransMat);
    subplot(plot_length,plot_length,NUM_PPT+1)
    g = digraph(avgTransMat);
    LWidth = 5*g.Edges.Weight/max(g.Edges.Weight);
    plot(g,'EdgeLabel',g.Edges.Weight,'LineWidth',LWidth, 'Layout', ...
        'circle','MarkerSize',10,'NodeColor','red', 'NodeLabel',{'On','MW','MB','??'})
    title('Average Across ppts')
end

%TODO: outdated
function f = ppt_transition_quarter_plot(labelMat,trans_func)
    %CAUTION: has to be divisible by 4
    %TODO: fix it
    QUARTER = size(labelMat,2)/4;
    for i = 1:4
        if i ==4
            label_quarter = labelMat(:,1+QUARTER*(i-1):end);
        else
            label_quarter = labelMat(:,1+QUARTER*(i-1):QUARTER*i+1);
        end
        f = ppt_transition_plot(labelMat,trans_func);
        set(f,'Name',sprintf('%d/4',i),'Numbertitle','off');
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions %%
%%%%%%%%%%%%%%%%%%%%%%


function f = transition_plot(transMat,title_name)
    % function to plot just one instance
    g = digraph(transMat);
    LWidth = 5*g.Edges.Weight/max(g.Edges.Weight);
    f=figure('name',title_name)
    plot(g,'EdgeLabel',g.Edges.Weight,'LineWidth',LWidth, 'Layout', ...
    'circle','MarkerSize',10,'NodeColor','red', 'NodeLabel',{'On','MW','MB','??'})
    title(title_name)
end

function transMat = calc_trans_prob(transMat)
    % function to calculate the probability based on each state instead 
    % of the average across all over
    transMat = transMat./sum(transMat,2);
    transMat(isnan(transMat)) = 0;
end

% Markov Chain ish
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

function avgTransMat = ppt_avg_transition(labelMat)
    NUM_PPT = size(labelMat,1);
    avgTransMat= zeros(4,4);
    NUM_PROBES = 0;
    for ppt = 1:NUM_PPT
        ppt_mat = labelMat(ppt,:); 
        ppt_mat = ppt_mat(not(isnan(ppt_mat))); % removing NaN
        NUM_PROBES = NUM_PROBES + length(ppt_mat);
        for i = 1:length(ppt_mat)-1
            avgTransMat(ppt_mat(i),ppt_mat(i+1)) = avgTransMat(ppt_mat(i),ppt_mat(i+1)) + 1;
        end
    end
    avgTransMat = avgTransMat/NUM_PROBES;
end