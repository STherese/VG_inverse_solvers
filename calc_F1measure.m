function [F1measure,TP,FP]=calc_F1measure(M,MTrue)
% Description:      Calculates the F1-measure
%
% Input:            M:          Estimated activation states
%                   M0:         True activation states
% Output:           F1measure:  The source retrieval index
%                   TP:         Number of true positives
%                   FP:         Number of false positives
%--------------------------References--------------------------------------
% Makhoul, J., Kubala, F., Schwartz, R., & Weischedel, R. (1999).
% Performance measures for information extraction. In Proceedings of DARPA
% Broadcast News Workshop (pp. 249–252).
%-----------------------------Author---------------------------------------
% Sofie Therese Hansen, DTU Compute
% March 2016
% -------------------------------------------------------------------------

actEst = find(M>0.5);
actTrue = find(MTrue);
TP = length(intersect(actEst,actTrue));
FP = length(actEst)- TP;
F1measure = 2*TP./(length(actTrue)+TP+FP);