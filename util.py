#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import glob
import random
import os.path as op
import cPickle as pickle
import numpy as np

def softmax(x):
    """
    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.
    """
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x

def sigmoid(x):
    s = 1.0 / (1 + np.exp(-x))
    return s

def sigmoid_grad(s):
    ds = s * (1 - s)
    return ds

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10, ANNEAL_EVERY = 20000, SAVE_PARAMS_EVERY = 5000, expcost=None):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    x = x0
    if not postprocessing:
        postprocessing = lambda x: x
    for iter in range(start_iter + 1, iterations + 1):
        cost = None
        cost, grad = f(x)
        x -= step * grad
        postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    return x