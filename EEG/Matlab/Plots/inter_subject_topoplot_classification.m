%%%%%%%%%%%%%
% Edit Here %
%%%%%%%%%%%%%
% *Do not put anything except the data under the directory
% Variables here
var_dir     = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/PreProcessed/Laplacian/Classification_Value';

% file, util. Has to have util.m in it. 
util_dir     = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/';

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

%%
%%%%%%%%%%
% Script %
%%%%%%%%%%

var_files = util('getBehaviorFiles',var_dir);
var_files = var_files(contains(var_files,'channels'));

lay=lay_BV64;
labels=lay.labels;
pos=lay.pos;

load(var_files{1})
ppts_mean_total = squeeze(linear_mean);


for f = 2:length(var_files)
    load(var_files{f})
    [~,name,~] = fileparts(var_files{f});
    ppts_mean_total = [ppts_mean_total;squeeze(linear_mean)];
end

plot_mean = squeeze(mean(ppts_mean_total));
disp(size(plot_mean))
plot_std  = std(ppts_mean_total);
plot_tscore = ones(1,64);
for i = 1:64
    [~,~,~,tstats] = ttest(ppts_mean_total(:,i),ones(size(ppts_mean_total(:,i)))*0.5);
    plot_tscore(1,i) = tstats.tstat;
end
names = {'Mean','std','t score'};
ind = 1;

f = figure('name', 'interSubject_topoplot_accuracy');
for values = [plot_mean; plot_std;plot_tscore]'
    subplot(1,3,ind)
    simpleTopoPlot2(values, pos', labels,2,[],0,lay,[]);
    title([names{ind},'(Laplacian, Linear SVM, 20 seconds bin)'])
    set(gca,'FontSize',18)
    if ind == 1
        colorbar; caxis([0.5 max(values)]);
    else
        colorbar;
    end
    ind = ind+1;
end


set(f, 'Position', [100, 100, 2000, 1500])
