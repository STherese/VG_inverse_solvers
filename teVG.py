"""
Created on Thu Aug 30 11:13:21 2018

 Description:      Solves the augmented inverse problem: y = A * s * x
                   using the Variational Garrote (VG).

 Input:            y:  Data matrix of size Kx1
                   A:  Design matrix/forward model of size KxN
                   gamma:  Sparsity level
                   opts:   see 'Settings'

 Output:           x:  Feature vector of size Nx1
                   m:  Variational mean/expectation of the state 's'.
                       Can be thresholded at 0.5 to only keep sources with
                       high evidence.
                   v:  The solution matrix; V = x*m;
--------------------------References--------------------------------------
 The Variational Garrote was originally presented in
 Kappen, H. (2011). The Variational Garrote. arXiv Preprint
 arXiv:1109.0486. Retrieved from http://arxiv.org/abs/1109.0486
 and
 Kappen, H.J., & Gómez, V. (2014). The Variational Garrote. Machine
 Learning, 96(3), 269–294. doi:10.1007/s10994-013-5427-7
 
 
 Time-expanded Variational Garrote reference
 Hansen, S.T., Stahlhut, C. & Hansen, L.K. (2013). Expansion
 of the Variational Garrote to a Multiple Measurement Vectors Model. In
 Twelfth Scandinavian Conference on Artificial Intelligence. ed. / M.
 Jaeger. IOS Press, (pp. 105–114).

 teVG with gradient descent
 Hansen, S.T., & Hansen, L.K. (2015). EEG source reconstruction
 performance as a function of skull conductance contrast. In 2015 IEEE
 International Conference on Acoustics, Speech and Signal Processing
 (ICASSP) (pp. 827–831). IEEE. doi:10.1109/ICASSP.2015.7178085

-----------------------------Author---------------------------------------
Sofie Therese Hansen, DTU Compute
August 2018
-------------------------------------------------------------------------
"""
import numpy as np
import scipy as sp
import scipy.sparse

def teVG(A,Y,gamma):
    
    
    
    # Settings
    max_iter=10000 #  Maximum number of iterations
    eta0=1e-3 #Learning rate for interpolation
    updEta=1 # Update of learning rate
    eta_start = 200 #When to reset learning rate
    tol=1e-5
    
    K,N = A.shape
    K,T=Y.shape
    A_mean = A.mean(axis=0)
    A = A - A_mean[np.newaxis,:]
    Y_mean = Y.mean(axis=0)
    Y = Y - Y_mean[np.newaxis,:]
    eps=np.finfo(float).eps
    m0 = np.zeros((N,1))
    m0 = np.array([max(min(xx, 1-np.sqrt(eps)), np.sqrt(eps)) for xx in m0])
    m0 = m0.reshape(N,1)
    m=m0
    chi_nn =1/K*(np.sum(A**2,axis=0))
    chi_nn = chi_nn.reshape(N,1)
    sp_tmp= sp.sparse.spdiags(np.transpose(m/(1-m)/chi_nn), 0, N,N).toarray()
    C = np.identity(K)+(1/K)*np.dot(np.dot(A,sp_tmp),np.transpose(A))
    Yhat = np.linalg.solve(C,Y)#C \ y
    YhatY = Yhat*Y;
    beta = T*K/np.sum(YhatY);
    lambda_param = beta*Yhat;
    X = np.transpose(np.reshape(np.array([(1/(K*beta*(1-m)*chi_nn))*np.reshape(np.dot(np.transpose(A),lambda_param[:,xx]),(N,1)) for xx in range(T)]),(T,N)))
    X2=X**2
    eta = eta0
    term = 0;k = 0;f = np.zeros((max_iter,1));
    
    #Beta = NaN(max_iter,1);

    while k < max_iter and term==0:
        
        #m = np.array([max(min(xx, 1-np.sqrt(eps)), np.sqrt(eps)) for xx in m])
        dFdm = np.reshape((beta*K/2)*np.sum(np.dot((1-2*m)*chi_nn,np.ones((1,T)))*X2,axis=1),(N,1))-gamma+np.log(m/(1-m))-np.reshape(np.sum(X*(np.dot(np.transpose(A),lambda_param)),axis=1),(N,1))
        m = m -eta*dFdm;
        a1=1
        m = np.array([max(min(m[xx,0], 1-np.sqrt(eps)), np.sqrt(eps)) for xx in range(N)])
        m = m.reshape(N,1)
        sp_tmp= sp.sparse.spdiags(np.transpose(m/(1-m)/chi_nn), 0, N,N).toarray()
        C = np.identity(K)+(1/K)*np.dot(np.dot(A,sp_tmp),np.transpose(A))
        Yhat = np.linalg.solve(C,Y)#C \ y
        YhatY = Yhat*Y;
        beta = T*K/np.sum(YhatY);
        lambda_param = beta*Yhat;
        X = np.transpose(np.reshape(np.array([(1/(K*beta*(1-m)*chi_nn))*np.reshape(np.dot(np.transpose(A),lambda_param[:,xx]),(N,1)) for xx in range(T)]),(T,N)))
        X2=X**2
        Z=Y-(1/beta)*lambda_param
        f[k] = -T*K/2*np.log(beta/(2*np.pi))+beta/2*np.sum((Z-Y)**2)\
        +K*beta/2*np.sum(X2*np.dot(chi_nn*m*(1-m),np.ones((1,T))))\
        -gamma*np.sum(m)+N*np.log(1+np.exp(gamma))\
        +np.sum(m*np.log(m)+(1-m)*np.log(1-m))\
        +np.sum(lambda_param*(Z-np.dot(A,(np.dot(m,np.ones((1,T)))*X)))) # Free energy
        
        #check for convergence
        if k>5:
            if f[k]==f[k-5] and f[k]==f[k-1]:
                term=1
            elif k>100:
                if abs(f[k]-f[k-5])<tol and abs(f[k]-f[k-1])<tol:
                    term=1
        #update learning rate
        if term==0 and updEta==1 and k>2:
            if f[k]>f[k-1]:
                eta=eta/2
            elif f[k]<f[k-1]:
                eta=eta*1.1
        if k>eta_start and k<eta_start+350:
            eta=1e-3
        k = k+1
    
    free_energy=f[k-1]
    V = X*np.dot(m,np.ones((1,T)))
   
  


    return m,V,X,free_energy

def sigmoid1(x):
    return 1/(1+np.exp(-x));