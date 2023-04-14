# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from .utils import roulette
from .utils import mat2bin
from .utils import bin2mat
from .utils import cross
from .utils import variation


class GeneticAlgorithm:
    def __init__(self, 
                 func,
                 dims,
                 *,
                 xlim=None,
                 population=50,
                 variation=0.05,
                 percentage=0.5,
                 max_iter=100,
                 tol=1e-4,
                 float_length=64,
                 int_length=None,
                 slow_learn=True,
                 verbose = 0,
                 random_state=None
                 ):
        self.func = func
        self.dims = dims
        self.xlim = xlim
        self.population = population
        self.variation = variation
        self.percentage = percentage
        self.max_iter = max_iter
        self.tol = tol
        self.int_length = int_length
        self.float_length = float_length
        self.slow_learn=slow_learn
        self.verbose = verbose
        self.random_state = random_state
        self.x = None
        # init
        np.random.seed(self.random_state)
        if self.xlim is None:
            self.int_length = 10
            self.xlim = [[-1023]*self.dims, [1023]*self.dims]
            self.x = np.random.uniform(self.xlim[0], self.xlim[1], (self.population, dims))
        else:            
            self.int_length = len(bin(int(np.max(np.abs(xlim))))) - 2
            # init x
            self.x = np.random.uniform([xlim[0][0], xlim[1][0]], [xlim[0][1], xlim[1][1]], (self.population, dims))
        # attribute
        self.best_ = None
        self.minf_ = np.inf
        self.iter_ = None
        
    def fit(self, display=False, curve=False):
        hisf, slow_time = list(), 0
        for ite in range(self.max_iter):
            fval = self.func(self.x)
            # chose
            idx = roulette(fval, self.percentage)
            evx = self.x[idx, :]
            evx2bin = mat2bin(evx, self.int_length, self.float_length)
            # cross
            newx = cross(evx2bin, self.population)
            # variation
            varx = variation(newx, self.percentage)
            # refresh
            self.x = bin2mat(varx)
            self.x = self.condition(self.x, self.xlim)
            fmin = np.min(fval)
            idxf = np.argmin(fval)
            if fmin < self.minf_:
                self.minf_ = fmin
                self.best_ = self.x[idxf, :]
            hisf.append(fmin)
            self.iter_ = ite
            if self.verbose:
                if display:
                    print(f'iter = {ite}, fmin = {fmin}')
            if self.slow_learn:
                continue
            else:
                if ite > 1 and slow_time < 10:
                    if np.abs(hisf[-1] - hisf[-2]) < self.tol:
                        slow_time += 1
                if slow_time >= self.slow_learn:
                    break
        if curve is True:
            self.plot_curve(hisf)
        return self.minf_, self.best_
    
    def plot_curve(self, hisf):
        plt.plot(hisf)
        plt.xlabel('iterations')
        plt.ylabel('function value')
        plt.show()
        return None
    
    def condition(self, x, limit):
        c = np.zeros_like(x)
        for d in range(self.dims):
            tmp = x[:, d]
            tmp[tmp < limit[0][d]] = limit[0][d]
            tmp[tmp > limit[1][d]] = limit[1][d]
            c[:, d] = tmp
        return c








