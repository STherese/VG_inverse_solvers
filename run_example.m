% Run simple example comparing VG, teVG and MarkoVG
SNRdes=5;
K=50;N=500;
rep=1;
rng(rep) % Seed for reprodicibility, change for more examples
N0=randi([1 4],1);
[A,Y,X_true,s,IDX,SNR]=genData(randn(K,N),SNRdes,N0);
actTrue=find(X_true);
S=zeros(size(X_true,1),1);
S(IDX)=1;

% original VG
Gammas=VG_cross(A,Y);
for t=1:25,[Vvg(:,t),Mvg(:,t),Xvg] = VG(A,Y(:,t),Gammas(t));end
[F1measureVG(rep),TPVG(rep),FPVG(rep)] = calc_F1measure(Mvg,X_true);
[F1measureVG2(rep),TPVG2(rep),FPVG2(rep)] = calc_F1measure(sum(Mvg,2),S);

% teVG
[gamma_mean1,gamma_median] = teVGGD_wcross(A,Y); % find sparsity
[VteVG,XteVG,mteVG,Ffull] = teVGGD(A,Y,gamma_median);
[F1measureteVG(rep),TPteVG(rep),FPteVG(rep)] = calc_F1measure(repmat(mteVG,1,size(X_true,2)),X_true);
[F1measureteVG2(rep),TPteVG2(rep),FPteVG2(rep)] = calc_F1measure(mteVG,S);

% MarkoVG
tic;[q,q0,fCross] = MarkoVGGD_cross(A,Y);toc
[VmarkoVG,MmarkoVG,X,F] = MarkoVGGD(A,Y,q,q);
MarkoVGact = find(MmarkoVG>0.5);
[F1measureMarko(rep),TPMarko(rep),FPMarko(rep)] = calc_F1measure(MmarkoVG,X_true);
[F1measureMarko2(rep),TPMarko2(rep),FPMarko2(rep)] = calc_F1measure(sum(MmarkoVG,2),S);

figure,
subplot(1,4,1)
plot(X_true');title('True')
subplot(1,4,2)
plot(Vvg');title('VG')
subplot(1,4,3)
plot(VteVG'),title('teVG')
subplot(1,4,4)
plot(VmarkoVG'),title('MarkoVG')
drawnow