function [q,q0,fCross]=MarkoVGGD_cross(A,Y,opts)
%
% Description:      Performs cross-validation to estimate the parameters
%                   of the Variational Garrote with a Markov prior
%                   (MarkoVG) which solves the augmented inverse problem: 
%                   Y = A * S * X.
%
% Input:            Y:  Data matrix of size KxT, where T is the number of
%                       measurements.
%                   A:  Design matrix/forward model of size KxN
%                   opts: see 'Settings'
%
% Output:           q:  Estimated probability of changing from inactive to
%                       active state
%                   q0: Start q-value
%                   fCross: Free energy on validation sets
%--------------------------References--------------------------------------
% The Variational Garrote was originally presented in 
% Kappen, H. (2011). The Variational Garrote. arXiv Preprint
% arXiv:1109.0486. Retrieved from http://arxiv.org/abs/1109.0486
% and
% Kappen, H.J., & Gómez, V. (2014). The Variational Garrote. Machine
% Learning, 96(3), 269–294. doi:10.1007/s10994-013-5427-7
%
% Preliminary MarkoVG reference 
% Hansen, S.T., & Hansen, L. (2013). EEG Sequence Imaging: A Markov Prior
% for the Variational Garrote. Proceedings of the 3rd NIPS Workshop on
% Machine Learning and Interpretation in Neuroimaging 2013. Retrieved from
% http://orbit.dtu.dk/fedora/objects/orbit:127330/datastreams/file_61f34d92-2e60-4871-8a41-08f1d69f5c47/content
%-----------------------------Author---------------------------------------
% Sofie Therese Hansen, DTU Compute
% March 2016
% -------------------------------------------------------------------------


% Settings:
try min_gamma0 = log10(opts.min_gamma0); catch; min_gamma0 = -50; end; % Minimum gamma value
try max_gamma = opts.max_gamma; catch; max_gamma = -1.12; end; % Maximum gamma value
try n_gamma = opts.n_gamma; catch; n_gamma = 20; end; % Number of gamma values
try n_gamma0 = opts.n_gamma0; catch; n_gamma0 = 20; end; % Number of gamma values for finding min_gamma
try opts.max_iter = opts.max_iter; catch; opts.max_iter = 500; end; % Maximum number of iterations
try opts.beta_tol = opts.beta_tol; catch; opts.beta_tol = 1e-2; end; % Convergence criterium for beta
try opts.k_beta_conv = opts.k_beta_conv; catch; opts.k_beta_conv = 10; end; % Convergence criterium for beta
try opts.eta0 = opts.eta0; catch; opts.eta0 = 1e-3; end; % Learning rate for gradient descent
try Cf = opts.Cf; catch;  Cf = 4; end; % Number of folds in the cross-validation
try opts.fact = opts.fact; catch;  opts.fact = 0.9; end; % Factor in smoothness=-fact*sparsity

rng(5) % Seed for reproducibility
indices = crossvalind('Kfold', size(Y,1), Cf);

if isfield(opts,'min_gamma') ==0 % calculate min_gamma
    qspace = logspace(min_gamma0,max_gamma,n_gamma0);
    disp('Calculating gamma_min')
    Mcross = zeros(1,n_gamma0);
    pair = 0;term = 0;
    while term == 0 && pair<n_gamma0;
        pair = pair+1;
        q = qspace((pair));
        p = q;
        for cf=1:Cf
            A_train = A(indices~=cf,:);
            Y_train = Y(indices~=cf,:);
            [~,M] = MarkoVGGD(A_train,Y_train,p,q,opts);
            Mcross(cf,pair)=sum(sum(M));
        end
        if mean(Mcross(:,pair))>= 0.5; % Check for evidence of activation
            term = 1;
        end
        
    end
    
    q0 = qspace(pair);
    qspace = logspace(log10(q0),max_gamma,n_gamma0);
    
else
    disp('Using provided gamma_min')
    q0 = 10^(opts.min_gamma);
    qspace = logspace(opts.min_gamma,max_gamma,n_gamma0);
end
%%
disp('Performing cross-validation')
%mseval=Inf(Cf,n_gamma);
fCross = Inf(Cf,n_gamma);
for pair = 1:n_gamma;
    fprintf('Running parameter pair %d of %d pairs\n',pair,n_gamma);
    q = qspace(pair);
    p = q;
    for kf = 1:Cf
        A_train = A(indices~=kf,:);
        opts.A_val = A(indices==kf,:);
        Y_train = Y(indices~=kf,:);
        opts.Y_val = Y(indices==kf,:);
        [~,~,~,~,Fval] = MarkoVGGD(A_train,Y_train,p,q,opts);
        fCross(kf,pair) = Fval;
        %mseval(kf,pair) = mean(mean((opts.A_val*(V)-opts.Y_val).^2));
    end
end

%%
[~, idx] = min((fCross(:,1:n_gamma)),[],2);
idx = floor(median(idx));
q = qspace(idx);
if idx == n_gamma
    disp('max_gamma was chosen')
end
