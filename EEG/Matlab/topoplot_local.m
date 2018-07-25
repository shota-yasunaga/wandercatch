%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% *Do not put anything except the data under the directory
% Variables here
var_dir     = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/accuracy';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

% Folder to save the epoch information .txt file
saving_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/Topoplot';
%%%%%%%%%%%%%%%%%%%%
% End of Editting  %
%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%
% Dependency %
%%%%%%%%%%%%%%
%util
addpath(util_dir)

addpath('/Users/macbookpro/Dropbox/College/TsuchiyaLab/LSCPtools/plot')
load('/Volumes/SHard/Tsuchiya_Lab_Data/myLayout_BV64.mat')



var_files = util('getBehaviorFiles',var_dir);


lay=lay_BV64;
labels=lay.labels;
pos=lay.pos;

for file = var_files
    load(file{1})
    [~,name,~] = fileparts(file{1});
    f = figure('name', [name(1:4), '_topoplot_accuracy']);
    set(f, 'Position', [100, 100, 2000, 1500])


    names = {'Mean','std','t score'};
    ind = 1;
    for values = [Linear_mean;Linear_std;Linear_tscore]'
        subplot(1,3,ind)
        simpleTopoPlot2(values, pos', labels,2,[],0,lay,[]);
        title([names{ind}, '(Linear Classifier)'])
        if ind == 1
            colorbar; caxis([0.4 0.75]);
        else
            colorbar;
        end
        ind = ind+1;
    end
    saveas(f, fullfile(saving_dir, [name, '(Linear Classifier)']), 'png'); % save it
end