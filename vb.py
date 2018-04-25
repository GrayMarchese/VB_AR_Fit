import numpy as np
from numpy import linalg, trace as tr

def noise_prec_params(n, w, X, prec, noise_b, noise_c):
    post_noise_b = 1/(0.5*linalg.norm(w)**2 + 0.5*tr(prec*X.transpose()*X)+ 1/noise_b)
    post_noise_c = (n/2 + noise_c)
    return post_noise_b, post_noise_c

def weight_prec_params(p, w, prec, weight_b, weight_c):
    post_weight_b = 1/(0.5*linalg.norm(w)**2 + 0.5*tr(prec)+ 1/weight_b)
    post_weight_c = (p/2 + weight_c)
    return post_weight_b, post_weight_c

def post_params(X, Y, w, prec, weight_b, weight_c, noise_b, noise_c):
    post_noise_b, post_noise_c = noise_prec_params(X.shape[0],w,X,prec,noise_b,noise_c)
    post_noise_prec = post_noise_b*post_noise_c
    post_weight_b, post_weight_c = weight_prec_params(X.shape[1],w,prec,weight_b,weight_c)
    post_weight_prec = post_weight_b*post_weight_c
    post_prec = linalg.inv(post_noise_prec*X.transpose()*X + post_weight_prec*np.identity(X.shape[1]))
    post_w = post_prec*X.transpose()*post_noise_prec*Y
    return post_w, post_prec, post_weight_b, post_weight_c, post_noise_b, post_noise_c

def variational_bayes()