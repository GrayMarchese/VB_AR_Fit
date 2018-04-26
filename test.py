from ar import *
from vb import *

N = 30000
True_W = [.2,0,-.4,-.2]
Noise_Variance = 1

ts = generate_ar(N,True_W,Noise_Variance,500)


print('\nTrue Weights: \n\t', True_W)

W, s = least_squares(ts,4)
print('\nMaximum Liklihood Estimates: \n\t', W)


order = 4
chunking = 50
W, updates, F = test_vb_ar_fit(ts,order,chunking)
print('\nVariational Bayesian Estimates (in %d updates):\n\t'%(updates), W)
print('\tFree Energy: ' + str(F))