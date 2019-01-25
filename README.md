# VG_inverse_solvers
Matlab: Implemented versions of Kappen's Variational Garrote and extended versions; teVG and MarkoVG. 
The teVG is an MMV version of VG and MarkoVG contains a Markov prior on the temporal support.  
run_example.m compares the methods.

Python: Implemented version of teVG and code for testing the algorithm.

VG reference:  
Kappen, H. J., & Gómez, V. (2014). The variational garrote. Machine Learning, 96(3), 269-294.

teVG reference:  
Hansen, S. T., Stahlhut, C., & Hansen, L. K. (2013). Expansion of the variational garrote to a multiple measurement vectors model. In 12th Scandinavian Conference on Artificial Intelligence (SCAI 2013) Scandinavian Conference on Artificial Intelligence (pp. 105-114). IOS Press.

teVG with gradient descent:  
Hansen, S. T., & Hansen, L. K. (2015, April). EEG source reconstruction performance as a function of skull conductance contrast. In Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on (pp. 827-831). IEEE.

MarkoVG reference:  
Hansen, S. T., & Hansen, L. K. (2017). Spatio-temporal reconstruction of brain dynamics from EEG with a Markov prior. NeuroImage, 148, 274-283.
