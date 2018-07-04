%%
clear all
close all
root_path='/Users/tand0009/Data/Wandercatch/Shota_Features/';

dir_ON=dir([root_path filesep 'ON' filesep 'freq*']);
dir_MW=dir([root_path filesep 'MW' filesep 'freq*']);

%%
Ch_Oz=30;
all_features_ON=[];
for nS=1:length(dir_ON)
    load([root_path filesep 'ON' filesep dir_ON(nS).name])
    fprintf('... %s trial n=%g\n',dir_ON(nS).name,size(features,1))
    all_features_ON(nS,:)=squeeze(mean(features(:,30,:),1));
    features_norm=features-repmat(mean(features(:,:,freqVec>0.2 & freqVec<45),3),[1 1 length(freqVec)]);
    all_features_ON_norm(nS,:)=squeeze(mean(features_norm(:,30,:),1));
    
    numTrials(nS,1)=size(features,1);
end

all_features_MW=[];
for nS=1:length(dir_MW)
    load([root_path filesep 'MW' filesep dir_MW(nS).name])
    fprintf('... %s trial n=%g\n',dir_MW(nS).name,size(features,1))
    all_features_MW(nS,:)=squeeze(mean(features(:,30,:),1));
    features_norm=features-repmat(mean(features(:,:,freqVec>0.2 & freqVec<45),3),[1 1 length(freqVec)]);
    all_features_MW_norm(nS,:)=squeeze(mean(features_norm(:,30,:),1));
 
    numTrials(nS,2)=size(features,1);
end

%%
figure;
subplot(1,2,1)
plot(freqVec,mean(all_features_ON(min(numTrials,[],2)>5,:)),'b')
hold on;
plot(freqVec,mean(all_features_MW(min(numTrials,[],2)>5,:)),'r')
xlim([0 50])

subplot(1,2,2)
plot(freqVec,mean(all_features_ON_norm(min(numTrials,[],2)>5,:)),'b')
hold on;
plot(freqVec,mean(all_features_MW_norm(min(numTrials,[],2)>5,:)),'r')
xlim([0 50])


[h pV]=ttest(all_features_ON(min(numTrials,[],2)>5,:),all_features_MW(min(numTrials,[],2)>5,:));
[h pV2]=ttest(all_features_ON_norm(min(numTrials,[],2)>5,:),all_features_MW_norm(min(numTrials,[],2)>5,:));

figure;
plot(freqVec,mean(all_features_ON(min(numTrials,[],2)>5,:))-mean(all_features_MW(min(numTrials,[],2)>5,:)),'b')
hold on
plot(freqVec,mean(all_features_ON_norm(min(numTrials,[],2)>5,:))-mean(all_features_MW_norm(min(numTrials,[],2)>5,:)),'r')
