% test best parameter for logSNR and Power computation
addpath(genpath(path_chronux))

SR=500;
dur=30;
clear data*
for nS=1:20
    for nTr=1:20
        signal = noise (dur, 1, SR)';
        signal=signal+0.5*sin((1:(SR*dur))/SR*12*2*pi);
        signal=signal+0.5*sin((1:(SR*dur))/SR*15*2*pi);
        data(nS,:,nTr)=signal;
%         flicker1(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*10*2*pi);
%         flicker2(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*12*2*pi);
%         flicker12(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*10*2*pi)+0.5*sin((1:(SR*dur))/SR*12*2*pi);
        
        
        for nW=1:5
            dataw(nTr,:,nW)=data(nS,(1:5*SR)+(nW-1)*SR,nTr);
        end
        data2(nS,:,nTr)=squeeze(mean(dataw(nTr,:,:),3))';
    end
end
param=[];
param.method='fft';
param.mindist=1;
[logSNR, faxis, logpow]=get_logSNR(data,SR,param);

% param=[];
% param.method='taper';
% param.numTaper=1;
% param.mindist=1;
[logSNR2, faxis2, logpow2]=get_logSNR(data2,SR,param);

[logSNR3, faxis3, logpow3]=get_logSNR(mean(data,3),SR,param);
[logSNR4, faxis4, logpow4]=get_logSNR(abs(hilbert(mean(data,3))),SR,param);

%%
figure;
plot(faxis,squeeze(mean(mean(logSNR,3),1)),'k');
hold on
% plot(faxis2,squeeze(mean(logSNR2,3)),'r');
% plot(faxis3,squeeze(mean(mean(logSNR3,3),1)),'b');
% plot(faxis4,squeeze(mean(mean(logSNR4,3),1)),'g');
xlim([0.5 25])

% [~,idx]=findclosest(faxis2,10); squeeze(mean(logSNR2(:,idx,:),3))/std(squeeze(mean(logSNR2(:,:,:),3)))
% [~,idx]=findclosest(faxis,10); squeeze(mean(logSNR(:,idx,:),3))/std(squeeze(mean(logSNR(:,:,:),3)))
