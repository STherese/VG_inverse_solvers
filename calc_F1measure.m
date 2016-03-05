function [F1measure,TP,FP]=calc_F1measure(M,MTrue)
actEst=find(M>0.5);
actTrue=find(MTrue);
TP=length(intersect(actEst,actTrue));
FP=length(actEst)- TP;
F1measure= 2*TP./(length(actTrue)+TP+FP);