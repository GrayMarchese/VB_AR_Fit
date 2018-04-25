from ar import *
from vb import *

ts = generate_ar(300,[.2,0,-.4,-.2],1,500)
W, s = least_squares(ts,4)
print('Maximum Liklihood Estimates: \n', W, '\n', s)


noise_b = 1000
noise_c = 0.001
weight_b = 1000
weight_c = 0.001
prec = s*np.identity(4)
X, Y = get_data(ts,4)
for iteration in range(4000):
    W, prec, weight_b, weight_c, noise_b, noise_c = post_params(X,Y,W,prec,weight_b,weight_c,noise_b,noise_c)

print('Variational Bayesian Estimates: \n', W, '\n', 1/(noise_b*noise_c))
