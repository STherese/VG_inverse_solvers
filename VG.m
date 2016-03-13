function [v,m,x] = VG(A,y,gamma,opts)
%
% Description:      Solves the augmented inverse problem: y = A * s * x
%                   using the Variational Garrote (VG).
%
% Input:            y:  Data matrix of size Kx1
%                   A:  Design matrix/forward model of size KxN
%                   gamma:  Sparsity level
%                   opts:   see 'Settings'
%
% Output:           x:  Feature vector of size Nx1
%                   m:  Variational mean/expectation of the state 's'.
%                       Can be thresholded at 0.5 to only keep sources with
%                       high evidence.
%                   v:  The solution matrix; V = x.*m;
%--------------------------References--------------------------------------
% The Variational Garrote was originally presented in
% Kappen, H. (2011). The Variational Garrote. arXiv Preprint
% arXiv:1109.0486. Retrieved from http://arxiv.org/abs/1109.0486
% and
% Kappen, H.J., & Gómez, V. (2014). The Variational Garrote. Machine
% Learning, 96(3), 269–294. doi:10.1007/s10994-013-5427-7
%
%-----------------------------Author---------------------------------------
% Sofie Therese Hansen, DTU Compute
% March 2016
% -------------------------------------------------------------------------

% Settings
try max_iter = opts.max_iter; catch; max_iter = 1000; end; % Maximum number of iterations
try eta0 = opts.eta0; catch; eta0=0.5; end; % Learning rate for interpolation
try updEta = opts.updEta; catch; updEta = 1; end; % Update of learning rate
try k_beta_conv = opts.k_beta_conv; catch; k_beta_conv = 100; end; % Convergence criterium for beta
try beta_tol = opts.beta_tol; catch; beta_tol = 1e-3; end; % Convergence criterium for beta

sigmoid1 = @(x) 1./(1+exp(-x));
[K,N] = size(A);
A = center(A);
y = center(y);
chi_nn = 1/K*sum(A.^2)';
m = zeros(N,1);
eta = eta0;
term = 0;
Beta = NaN(max_iter,1);
k = 0;
while k < max_iter && term==0
    k = k+1;
    m = max(m,sqrt(eps));
    m = min(m,1-sqrt(eps));
    C = eye(K)+1/K*A*spdiags(m./(1-m)./chi_nn,0,N,N)*A';
    yhat = C\y;
    yhaty = yhat.*y;
    beta = K/sum(yhaty(:));
    lambda = beta*yhat;
    x = (1./(K*beta*(1-m).*chi_nn)).*(A'*lambda);
    x2 = x.^2;
    mold = m;
    m = (1-eta)*mold + eta*sigmoid1(beta*K/2*x2.*chi_nn+gamma);
    if updEta==1 && max(abs(m-mold))>0.1
        eta = eta/2;
    end
    Beta(k) = beta;
    if k>k_beta_conv
        if abs(Beta(k)-Beta(k-5))<beta_tol && abs(Beta(k)-Beta(k-1))<beta_tol
            term = 1;
        end
    end
end
v = m.*x;