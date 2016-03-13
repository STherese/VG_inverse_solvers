function [Gammas] = VG_cross(A,Y,opts)
%
% Description:      Performs cross-validation to estimate the sparsity
%                   parameter gamma in Kappen et al.'s Variational
%                   Garrote (VG) which solves the augmented inverse
%                   problem: Y = A * S * X.
%
% Input:            Y:  Data matrix of size KxT, where T is the number of
%                       measurements.
%                   A:  Design matrix/forward model of size KxN
%                   opts: see 'Settings'
%
% Output:           Gammas: Sparsity levels for each T samples
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


% Settings:
try min_gamma = opts.min_gamma; catch; min_gamma = -100; end; % Minimum gamma value
try max_gamma = opts.max_gamma; catch; max_gamma = -1; end; % Maximum gamma value
try n_gamma = opts.n_gamma; catch; n_gamma = 20; end; % Number of gamma values
try opts.max_iter = opts.max_iter; catch; opts.max_iter = 500; end; % Maximum number of iterations
try opts.eta0 = opts.eta0; catch; opts.eta0=0.5; end; % Learning rate for interpolation
try opts.updEta = opts.updEta; catch; opts.updEta = 1; end; % Update of learning rate
try Cf = opts.Cf; catch;  Cf=4; end; % Number of folds in the cross-validation
try opts.k_beta_conv = opts.k_beta_conv; catch; opts.k_beta_conv = 10; end; % Convergence criterium for beta


%%
K = size(A,1);
T = size(Y,2);
gamma_all = linspace(min_gamma,max_gamma,n_gamma);
Gammas = NaN(T,1);

rng (5);
[indices] = crossvalind('Kfold',K,Cf);

for t = 1:T;
    error_val = NaN(Cf,n_gamma);
    fprintf('Calculating for time %d of %d\n',t,T);
    for cf = 1:Cf
        A_val  = center(A(indices==cf,:));
        y_val = center(Y(indices==cf,t));
        A_train = A(indices~=cf,:);
        y_train = Y(indices~=cf,t);
        for i = 1:n_gamma;
            [v,~,~] = VG(A_train,y_train,gamma_all(i),opts);
            error_val(cf,i) = mean(mean((A_val*v-y_val).^2))/mean(mean(y_val.^2));
        end
        
    end
    [~, imin] = min(error_val,[],2);
    gamma_mean = mean(gamma_all(imin));
    Gammas(t) = gamma_mean;
end
