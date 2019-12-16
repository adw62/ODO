#!/usr/local/bin/env python

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class Optimize(object):
    def __init__(self, data, net, seed_vec, target):
        self.bounds = get_bounds(data)
        scalerx, scalery = data.get_scaler()

        exp = scalery.transform([[target]])[0]
        x = scalerx.transform([seed_vec])

        fprime = lambda x, exp, net: approx_fprime(x, f, 0.01, exp, net)

        sol = minimize(f, x, args=(exp, net), jac=fprime, method='L-BFGS-B', bounds=self.bounds)
        self.property = scalery.inverse_transform([[float(net(sol.x))]])[0]
        self.solution = scalerx.inverse_transform([sol.x])[0]

def f(x, exp, net):
    #Here call to model
    fx = net(x).cpu().detach().numpy()
    return sum([(fxi-expi)**2 for fxi, expi in zip(fx, exp)])

def get_bounds(data):
    scalerx, scalery = data.get_scaler()
    tmp_data = data.x.cpu().detach().numpy()
    tmp_data = tmp_data.transpose()
    upper = [[max(x) for x in tmp_data]]
    lower = [[min(x) for x in tmp_data]]
    upper = scalerx.transform(upper)
    lower = scalerx.transform(lower)
    return [[x, y] for x, y in zip(lower[0], upper[0])]
        
        
