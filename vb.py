import numpy as np
from ar import least_squares
from numpy import linalg, trace as tr
from math import log, pi, ceil
from scipy.special import loggamma, digamma

def vb_ar_update(X, Y, w, cov, weight_b, weight_c, noise_b, noise_c):

    n = X.shape[0]
    p = X.shape[1]

    # Upper case is used for the posterior values and data.
    # Lower case is used for the priors and constants.

    # Weight and noise precision priors are converted to scalars
    weight_prec = np.asscalar(np.array(weight_b*weight_c))
    noise_prec = np.asscalar(np.array(noise_b*noise_c))

    # Update the weight precision gamma posterior parameters
    WEIGHT_B = 1/( 0.5*w.transpose()*w + 0.5*tr(cov) + 1/weight_b )
    WEIGHT_C = p/2 + weight_c
    WEIGHT_PREC =  np.asscalar(np.array(WEIGHT_B*WEIGHT_C))

    # Update the weight covariance matrix
    COV = linalg.inv(noise_prec*X.transpose()*X + WEIGHT_PREC*np.identity(p))
    
    # Update the weight vector mean
    W = COV*X.transpose()*noise_prec*Y

    # A term describing data variance and prediction error using priors
    # This is value is used in noise precision and convergence criteria
    E = 0.5*(Y - X*w).transpose()*(Y-X*w) + 0.5*tr(cov*X.transpose()*X)

    # Update the noise precision gamma posterior parameters
    NOISE_B = 1/( E + 1/noise_b )
    NOISE_C = n/2 + noise_c
    NOISE_PREC = NOISE_B*NOISE_C

    # VB Lower Bound
    beta = digamma(NOISE_C) + log(NOISE_B)
    lwr_bnd = n*beta/2 - NOISE_PREC*E  - n*log(2*pi)/2

    # Posterior||Prior KL divergences
    kl_w = kl_gaussian(W, COV, w, cov)
    kl_weight = kl_gamma(WEIGHT_B, WEIGHT_C, weight_b, weight_c)
    kl_noise = kl_gamma(NOISE_B, NOISE_C, noise_b, noise_c)

    # Negative Free-Energy
    F = lwr_bnd - kl_w - kl_weight - kl_noise

    return (W, COV, WEIGHT_B, WEIGHT_C, NOISE_B, NOISE_C, F)

def kl_gaussian( MU, COV, mu, cov):
    d = mu.shape[0]
    cov_inv = linalg.inv(cov)
    kl_div = 0.5*log(linalg.det(cov)/linalg.det(COV))
    kl_div = kl_div + 0.5*tr(cov_inv*COV)
    kl_div = kl_div + 0.5*(mu - MU).transpose()*cov_inv*(mu - MU)
    kl_div = kl_div - 0.5*d
    return kl_div

def kl_gamma(B,C,b,c):
    kl_div = (C-1)*digamma(C) - log(B) - C - loggamma(C) 
    kl_div = kl_div + loggamma(c) + c*log(b)
    kl_div = kl_div - (c -1)*(digamma(C) + log(B)) + (B*C)/b
    return kl_div

def test_vb_ar_fit(ts, order, chunking):

    N = len(ts)

    noise_b = 100
    noise_c = 0.001
    weight_b = 100
    weight_c = 0.001
    prec = np.identity(order)

    x = []
    y = []
    for i in range(len(ts)-order):
        x.append(ts[i:i+order][::-1])
        y.append(ts[i+order])
    X_full = np.matrix(x)
    Y_full = np.matrix(y).transpose()

    i = 0
    F_prev = 0.001
    while i < N:
        X = X_full[i:(i+chunking),:]
        Y = Y_full[i:(i+chunking)]
        if i == 0 :
            W, s = least_squares(ts[:(order + chunking)],order)
            print('\nInitial Prior: \n\t', W)
            W = W.transpose()
            prec = prec*s
        else:
            updated = vb_ar_update(X,Y,W,prec,weight_b,weight_c,noise_b,noise_c)
            W, prec, weight_b, weight_c, noise_b, noise_c, F = updated
            if abs(F - F_prev)/abs(F_prev) < 0.01:
                break
            F_prev = F
        i = i + chunking

    return W.transpose()[0], ceil((i+1)/chunking), np.asscalar(np.real(F))
