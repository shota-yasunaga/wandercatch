%%
clear all;
% close all;
run ../../localdef_wanderlust.m
addpath(genpath(path_spm12));
addpath(genpath(path_LSCPtools))
addpath(genpath(path_chronux))

files=dir([preproc_path filesep 'pPR_ffefspm_S*.mat']);
my_chan_ref=[17 22];

%%
all_logSNR_ON=[];
all_logSNR_MW=[];
for nS=1:length(files)
    filename=files(nS).name;
    subID=filename(findstr(filename,'spm_')+4:findstr(filename,'.')-1);
    fprintf('... %s\n',subID)
    
    % load behavioural data
    filebehav=dir([behav_path filesep 'wanderlust_behavres_s' subID(2:end) '*.mat']);
    Behav=load([behav_path filesep filebehav.name]);
    if nS==1
        GroupSubj(nS)=15;
    else
        GroupSubj(nS)=Behav.SubjectInfo.FlickerTG;
    end

    % load EEG data
    D=spm_eeg_load([preproc_path filesep 'pPR_ffefspm_' subID]);
    
    theseprobes=Behav.all_probe_responses;
    if strcmp(subID,'S13')
        fprintf('... ... correcting for wrong key-board mapping\n')
        right=[84 85 86 86];
        wrong=[11 12 13 14];
        for m=1:4
            theseprobes(theseprobes(:,6)==wrong(m),9)=m;
        end
    end
    my_probes=[];
    for nprobe=1:max(theseprobes(:,1))
        nB=unique(theseprobes(theseprobes(:,1)==nprobe,2));
        nBt=unique(theseprobes(theseprobes(:,1)==nB,4));
        nBn=unique(theseprobes(theseprobes(:,1)==nB,3));
        
        this_probe=theseprobes(theseprobes(:,1)==nprobe,[5 9]);
        my_probes=[my_probes ; [nprobe nB nBt nBn  this_probe(:,2)']];
    end
    
    bef_eeg=D(1:64,D.indsample(-10):D.indsample(0)-1,:)-repmat(mean(D([29 31],D.indsample(-10):D.indsample(0)-1,:),1),64,1,1);
    aft_eeg=D(1:64,D.indsample(0)+1:D.indsample(10),:)-repmat(mean(D([29 31],D.indsample(0)+1:D.indsample(10),:),1),64,1,1);
    
    
    param=[];
    param.method='fft';
    param.mindist=1;
    [logSNR, faxis, logpow]=get_logSNR(bef_eeg,D.fsample,param);
    faxis2=faxis(faxis>0.5 & faxis<50);
    logSNR_bef(nS,:,:)=mean(logSNR(:,faxis>0.5 & faxis<50,:),3);
    logPow_bef(nS,:,:)=mean(logpow(:,faxis>0.5 & faxis<50,:),3);
    for nstate=1:3
        logSNR_bef_state(nS,nstate,:,:)=mean(logSNR(:,faxis>0.5 & faxis<50,my_probes(:,6)==nstate),3);
        logPow_bef_state(nS,nstate,:,:)=mean(logpow(:,faxis>0.5 & faxis<50,my_probes(:,6)==nstate),3);
        num_state(nS,nstate)=sum(my_probes(:,6)==nstate);
    end
      all_logSNR_ON=cat(3,all_logSNR_ON,logSNR(:,faxis>0.5 & faxis<50,my_probes(:,6)==1));
    all_logSNR_MW=cat(3,all_logSNR_MW,logSNR(:,faxis>0.5 & faxis<50,my_probes(:,6)==2));
  
    [logSNR, faxis, logpow]=get_logSNR(aft_eeg,D.fsample,param);
    logSNR_aft(nS,:,:)=mean(logSNR(:,faxis>0.5 & faxis<50,:),3);
    logPow_aft(nS,:,:)=mean(logpow(:,faxis>0.5 & faxis<50,:),3);
    for nstate=1:3
        logSNR_aft_state(nS,nstate,:,:)=mean(logSNR(:,faxis>0.5 & faxis<50,my_probes(:,6)==nstate),3);
        logPow_aft_state(nS,nstate,:,:)=mean(logpow(:,faxis>0.5 & faxis<50,my_probes(:,6)==nstate),3);
    end
    
%     param=[];
%     param.method='taper';
%     param.numTaper=1;
%     param.mindist=1;
%     [logSNR, faxis3, logpow]=get_logSNR(bef_eeg,D.fsample,param);
%     faxis4=faxis3(faxis3>0.5 & faxis3<50);
%     logSNR_bef2(nS,:,:)=mean(logSNR(:,faxis3>0.5 & faxis3<50,:),3);
%     logPow_bef2(nS,:,:)=mean(logpow(:,faxis3>0.5 & faxis3<50,:),3);
%     for nstate=1:3
%         logSNR_bef_state2(nS,nstate,:,:)=mean(logSNR(:,faxis3>0.5 & faxis3<50,my_probes(:,6)==nstate),3);
%         logPow_bef_state2(nS,nstate,:,:)=mean(logpow(:,faxis3>0.5 & faxis3<50,my_probes(:,6)==nstate),3);
%     end
%     
%     [logSNR, faxis3, logpow]=get_logSNR(aft_eeg,D.fsample,param);
%     logSNR_aft2(nS,:,:)=mean(logSNR(:,faxis3>0.5 & faxis3<50,:),3);
%     logPow_aft2(nS,:,:)=mean(logpow(:,faxis3>0.5 & faxis3<50,:),3);
%     for nstate=1:3
%         logSNR_aft_state2(nS,nstate,:,:)=mean(logSNR(:,faxis3>0.5 & faxis3<50,my_probes(:,6)==nstate),3);
%         logPow_aft_state2(nS,nstate,:,:)=mean(logpow(:,faxis3>0.5 & faxis3<50,my_probes(:,6)==nstate),3);
%     end
end

%%
figure;
plot(faxis2,squeeze(mean(all_logSNR_ON(30,:,:),3)),'b');
hold on
plot(faxis2,squeeze(mean(all_logSNR_MW(30,:,:),3)),'r')

%%
figure
subplot(4,2,1)
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,1,30,:),1)),'b');
hold on;
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,2,30,:),1)),'r')
xlim([0.5 40])
subplot(4,2,3)
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,1,30,:)-logPow_bef_state(num_state(:,2)>5,2,30,:),1)))
xlim([0.5 40])

% figure
subplot(4,2,5)
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,1,30,:),1)),'b')
hold on;
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,2,30,:),1)),'r')
xlim([0.5 40])
subplot(4,2,7)
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,1,30,:)-logSNR_bef_state(num_state(:,2)>5,2,30,:),1)))
xlim([0.5 40])
% %%
% figure
% subplot(4,2,2)
% plot(faxis4,squeeze(mean(logPow_bef_state2(num_state(:,2)>5,1,30,:),1)),'b')
% hold on;
% plot(faxis4,squeeze(mean(logPow_bef_state2(num_state(:,2)>5,2,30,:),1)),'r')
% xlim([0.5 40])
% subplot(4,2,4)
% plot(faxis4,squeeze(mean(logPow_bef_state2(num_state(:,2)>5,1,30,:)-logPow_bef_state2(num_state(:,2)>5,2,30,:),1)))
% xlim([0.5 40])

% % figure
% subplot(4,2,6)
% plot(faxis4,squeeze(mean(logSNR_bef_state2(num_state(:,2)>5,1,30,:),1)),'b')
% hold on;
% plot(faxis4,squeeze(mean(logSNR_bef_state2(num_state(:,2)>5,2,30,:),1)),'r')
% xlim([0.5 40])
% subplot(4,2,8)
% plot(faxis4,squeeze(mean(logSNR_bef_state2(num_state(:,2)>5,1,30,:)-logSNR_bef_state2(num_state(:,2)>5,2,30,:),1)))
% xlim([0.5 40])

%%
figure;
[h pV2]=ttest(squeeze(logPow_bef_state(num_state(:,2)>5,1,30,:)),squeeze(logPow_bef_state(num_state(:,2)>5,2,30,:)));
subplot(2,2,1)
plot(faxis2,pV2);
xlim([0.5 40])

% [h pV2]=ttest(squeeze(logPow_bef_state2(num_state(:,2)>5,1,30,:)),squeeze(logPow_bef_state2(num_state(:,2)>5,2,30,:)));
% subplot(2,2,2)
% plot(faxis4,pV2);
% xlim([0.5 40])

[h pV2]=ttest(squeeze(logSNR_bef_state(num_state(:,2)>5,1,30,:)),squeeze(logSNR_bef_state(num_state(:,2)>5,2,30,:)));
subplot(2,2,3)
plot(faxis2,pV2);
xlim([0.5 40])

% [h pV2]=ttest(squeeze(logSNR_bef_state2(num_state(:,2)>5,1,30,:)),squeeze(logSNR_bef_state2(num_state(:,2)>5,2,30,:)));
% subplot(2,2,4)
% plot(faxis4,pV2);
% xlim([0.5 40])

%%
figure
subplot(2,2,1)
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,1,:,:),1))','b');
hold on;
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])
subplot(2,2,2)
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,1,:,:),1))','b');
hold on;
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])

subplot(2,2,3)
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,1,:,:),1))'-squeeze(mean(logPow_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])
subplot(2,2,4)
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,1,:,:),1))'-squeeze(mean(logSNR_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])

%%
figure
subplot(2,2,1)
plot(faxis2,squeeze(mean(mean(logPow_bef_state(num_state(:,2)>5,1,:,:),3),1))','b');
hold on;
plot(faxis2,squeeze(mean(mean(logPow_bef_state(num_state(:,2)>5,2,:,:),3),1))','r');
xlim([0.5 40])
subplot(2,2,2)
plot(faxis2,squeeze(mean(mean(logSNR_bef_state(num_state(:,2)>5,1,:,:),3),1))','b');
hold on;
plot(faxis2,squeeze(mean(mean(logSNR_bef_state(num_state(:,2)>5,2,:,:),3),1))','r');
xlim([0.5 40])

subplot(2,2,3)
plot(faxis2,squeeze(mean(logPow_bef_state(num_state(:,2)>5,1,:,:),1))'-squeeze(mean(logPow_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])
subplot(2,2,4)
plot(faxis2,squeeze(mean(logSNR_bef_state(num_state(:,2)>5,1,:,:),1))'-squeeze(mean(logSNR_bef_state(num_state(:,2)>5,2,:,:),1))','r')
xlim([0.5 40])
