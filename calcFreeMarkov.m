function f=calcFreeMarkov(A,beta,Y,Ytilde,Q,M,X,X2,chi_nn)
% Calculates the free energy in MarkoVG
%
%-----------------------------Author---------------------------------------
% Sofie Therese Hansen, DTU Compute
% March 2016
% -------------------------------------------------------------------------


lambda = beta*Ytilde;
N = size(M,1);
z = Y-1./beta*lambda;
[K,T] = size(Y);
g00 = Q(1,1);
g11 = Q(2,2);
g01 = Q(2,1);
g10 = Q(1,2);

t = 2;
s1 = sum(M(:,1).*log(g10/g00));
s2 = sum(sum(M(:,t:end).*log(g10/g00)+M(:,t-1:end-1)+M(:,t:end).*M(:,t-1:end-1)*log(g00*g11/(g01*g10))));


f = -T*K/2*log(beta/(2*pi))+beta/2*sum(sum((z-Y).^2))...
    +K*beta/2*sum(sum(X2.*M.*(1-M).*repmat(chi_nn,1,T)))...
 + N*T*log(1/g00) - s1 -s2...
 +sum(sum(M.*log(M)+(1-M).*log(1-M)))...
    +sum(sum(lambda.*(z-A*((M).*X))));