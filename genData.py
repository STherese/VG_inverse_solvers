# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:52:29 2018

@author: sofha
"""

def genData(A,SNRdes,N0):
    import numpy as np
    from butter_fil import butter_bandpass_filter
    #K,N =np.ndarray.size(A);
    K,N =A.shape;
   
    IDX = np.random.randint(1,N,size=N0);
    print(IDX)
    fs = 200; # Sampling frequency
    T = 100; # T time samples
    maxFreq = np.random.randint(10, 15,N0);
    X_true = np.zeros((N,T));
    
    
    for s in range(N0):
        print(s)
        Sx = 0
        it=1
        #while len(np.nonzero(Sx))<2:
        Minidx=1
        #while (len(np.nonzero(Minidx))<2 or Minidx[end]-Minidx[0]<2) and it==1:
        Sx = np.random.randn(1,T+1000); # creates white noise
        #plt.figure(s)
        #plt.plot(np.transpose(Sx))
        #plt.show
        it=2
        Sx = butter_bandpass_filter(Sx, maxFreq[s]-1/2, maxFreq[s]+1/2, fs, order=6)
        #.plot(np.transpose(Sx))
        Sx=Sx[0,500:500+T]
                 
                #Minidx = find(abs(Sx)<(max(abs(Sx))*0.05));
            #Sx[0:Minidx[0]] = 0;
            #Sx[Minidx[end]:end] = 0; 
        X_true[IDX[s],:] = Sx;
    Y0 = np.dot(A,X_true);
    stdnoise = np.std(np.ravel(Y0))*10**(-SNRdes/20);
    noise = stdnoise*np.random.randn(K,T);
    Y=Y0+noise;
    SNR = 10*np.log10(np.mean(np.var(Y0))/np.mean(np.var(noise)));

    
    return Sx,A,IDX,X_true,SNR,Y