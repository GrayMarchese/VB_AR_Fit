from ar import *
from vb import *

ts = generate_ar(300,[.2,0,-.4,-.2],1,500)
W, s = least_squares(ts,4)
#print('Maximum Liklihood Estimates: \n', W, '\n', s)


noise_b = 10
noise_c = 0.01
weight_b = 10
weight_c = 0.1
prec = s*np.identity(4)
X, Y = get_data(ts,4)

print('N = '+ str(X.shape[0]))
F_prev = 0.001
for iteration in range(4):
    updated = vb_ar_update(X,Y,W,prec,weight_b,weight_c,noise_b,noise_c)
    W, prec, weight_b, weight_c, noise_b, noise_c, F = updated
    if abs(F - F_prev)/abs(F_prev) - 1 < 0.01 :
        print('Broke early at iteration'+str(iteration))
        break
    F_prev = F

#print('Variational Bayesian Estimates: \n', W, '\n', 1/(noise_b*noise_c))
