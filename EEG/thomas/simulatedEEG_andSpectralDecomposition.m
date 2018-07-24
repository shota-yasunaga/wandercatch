% test best parameter for logSNR and Power computation
run ../../localdef_wanderlust.m
addpath(genpath(path_chronux))
addpath(genpath(path_filedtrip))
SR=500;
dur=30;
clear data*
for nS=1:20
    for nTr=1:20
        signal = noise(dur, 1, SR)';
        signal=signal+0.5*sin((1:(SR*dur))/SR*12*2*pi);
        signal=signal+0.5*sin((1:(SR*dur))/SR*15*2*pi);
        data(nS,:,nTr)=signal;
        data3(nS,:,nTr)=[noise(5, 1, SR)' signal];
%         flicker1(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*10*2*pi);
%         flicker2(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*12*2*pi);
%         flicker12(1,:,nTr)=0.5*sin((1:(SR*dur))/SR*10*2*pi)+0.5*sin((1:(SR*dur))/SR*12*2*pi);
        
        
        for nW=1:5
            dataw(nTr,:,nW)=data(nS,(1:5*SR)+(nW-1)*SR,nTr);
        end
        data2(nS,:,nTr)=squeeze(mean(dataw(nTr,:,:),3))';
        
        % make wavelet for 10Hz with 10 cycles
        myfreqs=5:0.1:40;
        figure; hold on;
        xTime=(1/SR:1/SR:35)-5;
        for nf=1:length(myfreqs)
            tpx2=(1:(1/myfreqs(nf)*20*SR))/SR-(1/myfreqs(nf)*20)/2;
            sinewave=cos(2*pi*myfreqs(nf)*tpx2);
            length_wave(nf)=1/myfreqs(nf)*20;
            tpx=linspace(-4, 4, length(sinewave));
            gausswave=exp(-(tpx.^2)/2);
            mywavelet=sinewave.*gausswave;
            plot(tpx2,mywavelet)
             tf_decomp(nf,:,nTr)=abs(hilbert(conv(mean(data3(nS,:,nTr),3),mywavelet,'same')));
       
        end
         [faxis,pow]=get_PowerSpec((data3(nS,xTime>0,nTr)),SR,0,0); 
        pow_decomp(nTr,:)=pow;
    end
    myfreqs=5:0.1:40;
    figure; hold on;
    xTime=(1/SR:1/SR:35)-5;
    for nf=1:length(myfreqs)
        tpx2=(1:(1/myfreqs(nf)*20*SR))/SR-(1/myfreqs(nf)*20)/2;
        sinewave=cos(2*pi*myfreqs(nf)*tpx2);
        length_wave(nf)=1/myfreqs(nf)*20;
        tpx=linspace(-4, 4, length(sinewave));
        gausswave=exp(-(tpx.^2)/2);
        mywavelet=sinewave.*gausswave;
        plot(tpx2,mywavelet)
        tf_decomp2(nf,:)=abs(hilbert(conv(mean(data3(nS,:,:),3),mywavelet,'same')));
    end
        
    figure;
    subplot(2,2,1)
    [faxis,pow]=get_PowerSpec(mean(data3(nS,xTime>0,:),3),SR,0,0);
    plot(faxis,pow)
    xlim([1 30]);
    subplot(2,2,2)
    plot(myfreqs,mean(tf_decomp2(:,xTime>0),2))
    xlim([1 30]);
    subplot(2,2,3)
    plot(faxis,mean(pow_decomp,1))
      xlim([1 30]);
  subplot(2,2,4)
    plot(myfreqs,mean(mean(tf_decomp(:,xTime>0,:),2),3))
    xlim([1 30]);
%     imagesc((1/SR:1/SR:35)-5,myfreqs,tf_decomp2)
    
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
