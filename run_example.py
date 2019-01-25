# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:53:04 2018
Run simple example testing the teVG (time-expanded variational garrote).

Time-expanded Variational Garrote reference
Hansen, S.T., Stahlhut, C. & Hansen, L.K. (2013). Expansion
of the Variational Garrote to a Multiple Measurement Vectors Model. In
Twelfth Scandinavian Conference on Artificial Intelligence. ed. / M.
Jaeger. IOS Press, (pp. 105–114).

-----------------------------Author---------------------------------------
 Sofie Therese Hansen, DTU Compute
 August 2018
-------------------------------------------------------------------------

"""

from butter_fil import *
import numpy as np
import matplotlib.pyplot as plt
from genData import *
#import random

SNRdes = 5;
K = 50;N = 500;
rep = 1;
np.random.seed(rep)
N0=np.random.randint(1,4)
A=np.random.randn(K,N)
Sx,A,IDX,X_true,SNR,Y= genData(A,SNRdes,N0)

plt.figure(5)
plt.plot(np.transpose(X_true[IDX,:]))

#%% 
from teVG import *
gamma=-100 # usually found using cross-validation

m,V,X,free_energy=teVG(A,Y,gamma)
plt.figure(6)
plt.plot(np.transpose(V).T)
