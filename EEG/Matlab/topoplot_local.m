
load('/Volumes/SHard/Tsuchiya_Lab_Data/myLayout_BV64.mat')
load('/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/0.005-5-510Folds.mat')
addpath('/Users/macbookpro/Dropbox/College/TsuchiyaLab/LSCPtools/plot')

lay=lay_BV64;
labels=lay.labels;
pos=lay.pos;
figure;



names = {'Linear','Polynomial','rbf'};
ind = 1;
for temp = [linear;polynomial;rbf]'
    disp(length(temp))
    subplot(1,3,ind)
    simpleTopoPlot2(temp, pos', labels,2,[],0,lay,[]);
    title(['Accuracy of',names{ind} ,'Classifier'])
    colorbar; caxis([0.4 0.75]);
    ind = ind+1;
end


