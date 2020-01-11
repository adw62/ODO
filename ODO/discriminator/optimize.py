#!/usr/local/bin/env python

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class Optimize(object):
    def __init__(self, data, net, seed_vec, target):

        scalerx, scalery = data.get_scaler()
        exp = scalery.transform([[target]])[0]
        x = scalerx.transform([seed_vec])

        self.bounds = get_bounds(data, x[0], use_full_data_range=True)

        fprime = lambda x, exp, net: approx_fprime(x, f, 0.001, exp, net)

        sol = minimize(f, x, args=(exp, net), jac=fprime, method='L-BFGS-B', bounds=self.bounds)
        self.property = scalery.inverse_transform([[float(net(sol.x))]])[0]
        self.solution = scalerx.inverse_transform([sol.x])[0]

def f(x, exp, net):
    #Here call to model
    fx = net(x).cpu().detach().numpy()
    return sum([(fxi-expi)**2 for fxi, expi in zip(fx, exp)])


def get_bounds(data, seed, msd=0.01, use_full_data_range=True):
    tmp_data = data.x.cpu().detach().numpy()
    tmp_data = tmp_data.transpose()
    upper = [max(x) for x in tmp_data]
    lower = [min(x) for x in tmp_data]
    if not use_full_data_range:
        msd_upper = [x+msd for x in seed]
        msd_lower = [x-msd for x in seed]
        upper = [min([x, y]) for x, y in zip(upper, msd_upper)]
        lower = [max([x, y]) for x, y in zip(lower, msd_lower)]
    return [[x, y] for x, y in zip(lower, upper)]

