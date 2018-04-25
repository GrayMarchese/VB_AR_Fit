import numpy as np
from ar import least_squares
from numpy import linalg, trace as tr
from math import log, pi, ceil
from scipy.special import loggamma, digamma

def vb_ar_update(X, Y, w, prec, weight_b, weight_c, noise_b, noise_c):

    #Upper case is used for the posterior values and data.
    #Lower case is used for the priors and constants.
    n = X.shape[0]
    p = X.shape[1]

    #A value used in the noise posterior and convergence criteria
    E_D = 0.5*linalg.norm(w)**2 + 0.5*tr(prec*X.transpose()*X)

    #Update noise precision gamma posterior
    NOISE_B = 1/(E_D+1/noise_b)
    NOISE_C = (n/2 + noise_c)
    NOISE_PREC = NOISE_B*NOISE_C

    #Update weight precision gamma posterior
    WEIGHT_B = 1/(0.5*linalg.norm(w)**2 + 0.5*tr(prec)+ 1/weight_b)
    WEIGHT_C = (p/2 + weight_c)
    WEIGHT_PREC = WEIGHT_B*WEIGHT_C

    #Update wieght and covariance matrix posterior means.
    PREC = linalg.inv(NOISE_PREC*X.transpose()*X + WEIGHT_PREC*np.identity(p))
    W = PREC*X.transpose()*NOISE_PREC*Y

    #VB Lower Bound
    beta = digamma(NOISE_C) + log(NOISE_B)
    lwr_bnd = n*beta/2 - NOISE_PREC*E_D  - n*log(2*pi)/2

    #POSTERIOR/Prior KL divergences
    kl_w = kl_gaussian(W, PREC, w, prec)
    kl_weight = kl_gamma(WEIGHT_B, WEIGHT_C, weight_b, weight_c)
    kl_noise = kl_gamma(NOISE_B, NOISE_C, noise_b, noise_c)

    #Negative Free-Energy

    # print('lower bound: ' + str(type(lwr_bnd)))
    # print('weight: ' + str(type(kl_w)))
    # print('weight prec: ' +  str(type(kl_weight)))
    # print('noise prec: ' + str(type(kl_noise)))

    F = lwr_bnd - kl_w - kl_weight - kl_noise

    return (W, PREC, WEIGHT_B, WEIGHT_C, NOISE_B, NOISE_C, F)

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

    noise_b = 1000
    noise_c = 0.001
    weight_b = 1000
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
            W = W.transpose()
            prec = prec*s
        else:
            updated = vb_ar_update(X,Y,W,prec,weight_b,weight_c,noise_b,noise_c)
            W, prec, weight_b, weight_c, noise_b, noise_c, F = updated
            if i > 60 :
                break
            F_prev = F
        i = i + chunking

    return W.transpose()[0], ceil((i+1)/3), F.item(0)
