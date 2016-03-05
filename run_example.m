for rep=1:25
SNRdes=5;
K=50;N=500;
rng(rep) % Seed for reprodicibility, change for more examples
N0=randi([1 4],1);
[A,Y,X_true,s,IDX,SNR]=genData(randn(K,N),SNRdes,N0);
actTrue=find(X_true);
S=zeros(size(X_true,1),1);
S(IDX)=1;

% original VG
%addpath C:\Users\sofha\Dropbox\PhD\Afhandling\matlab
%[v_vg,m_vg,Gammas] = dual_timeThesis(G,x,20);
Gammas=VG_cross(A,Y);
for t=1:25,[Vvg(:,t),Mvg(:,t),Xvg] = VG(A,Y(:,t),Gammas(t));end
[F1measureVG(rep),TPVG(rep),FPVG(rep)] = calc_F1measure(Mvg,X_true);
[F1measureVG2(rep),TPVG2(rep),FPVG2(rep)] = calc_F1measure(sum(Mvg,2),S);

% teVG
[gamma_mean1,gamma_median] = teVGGD_wcross2(A,Y); % find sparsity
[VteVG,XteVG,mteVG,Ffull] = teVGGD(A,Y,gamma_median);
[F1measureteVG(rep),TPteVG(rep),FPteVG(rep)] = calc_F1measure(repmat(mteVG,1,size(X_true,2)),X_true);
[F1measureteVG2(rep),TPteVG2(rep),FPteVG2(rep)] = calc_F1measure(mteVG,S);
% Marko
%addpath C:\Users\sofha\MarkoVGGD
%[p0]=MarkoVGGD0(G,x,20);
%[p1,q1,m1,w1,Mcross1,idx01,p01]=MarkoVGGD0Thesis(G,x,20,3);
%idx0=16;
%[p,q,m_marko,w_marko,fCross,qspace,f]=MarkoVGGDornqFactMedianThesis(G,x,eye(500),20,0.9,3,0,p0);
%p=8.0156e-09
%[V,M,X,F]=MarkoVGGD2(G,x,p,q,opts);

tic;[q,q0,fCross] = MarkoVGGD_cross(A,Y);toc
[VmarkoVG,MmarkoVG,X,F] = MarkoVGGD(A,Y,q,q);
MarkoVGact = find(MmarkoVG>0.5);
[F1measureMarko(rep),TPMarko(rep),FPMarko(rep)] = calc_F1measure(MmarkoVG,X_true);
[F1measureMarko2(rep),TPMarko2(rep),FPMarko2(rep)] = calc_F1measure(sum(MmarkoVG,2),S);

figure,
subplot(1,4,1)
plot(X_true')
subplot(1,4,2)
plot(Vvg')
subplot(1,4,3)
plot(VteVG')
subplot(1,4,4)
plot(VmarkoVG')
drawnow
end