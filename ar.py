import numpy as np
from math import sqrt
from numpy import linalg
from numpy.random import normal 

def generate_ar(n, weights, noise_var, burn_in):
    ts = []
    for i in weights:
        ts.append(0)
    order = len(weights)
    for t in range(n-order+burn_in):
        x_t = 0
        for i in range(1,order+1):
            x_t = x_t + ts[-i]*weights[i-1]
        x_t = x_t + normal(0,sqrt(noise_var),1)[0]
        ts.append(x_t)
    return ts[burn_in:]

def get_data(ts,order):
    x = []
    y = []
    for i in range(len(ts)-order):
        x.append(ts[i:i+order][::-1])
        y.append(ts[i+order])
    X = np.matrix(x)
    Y = np.matrix(y).transpose()
    return X, Y

def least_squares(ts, order):
    X, Y = get_data(ts, order)
    W = linalg.inv(X.transpose()*X)*X.transpose()*Y
    S = (1/(len(ts)-order-1))*(linalg.norm(X*W - Y)**2)
    return W.transpose()[0], S