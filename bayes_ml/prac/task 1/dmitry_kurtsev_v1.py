# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import binom, poisson


import numpy as np
from scipy.stats import binom, poisson


def pa(params, model):
    n = params['amax'] - params['amin'] + 1
    return np.ones(n) / n, np.arange(params['amin'], params['amax'] + 1)


def pb(params, model):
    n = params['bmax'] - params['bmin'] + 1
    return np.ones(n) / n, np.arange(params['bmin'], params['bmax'] + 1)


def pc_ab(a, b, params, model):
    cmax = params['amax'] + params['bmax'] + 1
    vals = np.arange(cmax)
    p1, p2 = params['p1'], params['p2']

    if model == 1:
        a_probs = binom.pmf(np.arange(cmax), a.reshape(-1, 1), p1)
        b_probs = binom.pmf(np.arange(cmax), b.reshape(-1, 1), p2)
        probs = np.zeros((cmax, len(a), len(b)))
        for k in range(cmax):
            probs[k] = np.dot(a_probs[..., :k + 1], b_probs[..., :k + 1][..., ::-1].T)
        return probs, vals

    if model == 2:
        lambda_pois = p1 * a.reshape(-1, 1) + p2 * b
        probs = poisson.pmf(vals, np.expand_dims(lambda_pois, axis=2))[..., :cmax].transpose((2, 0, 1))
        return probs, vals


def pc(params, model):
    a = np.arange(params['amin'], params['amax'] + 1)
    b = np.arange(params['bmin'], params['bmax'] + 1)
    pc_ab_probs, _ = pc_ab(a, b, params, model)
    pc_ab_probs = pc_ab_probs 
    pc_probs = pc_ab_probs.sum(axis=(1, 2)).reshape(-1,) / len(b) / len(a)
    vals = np.arange(params['amax'] + params['bmax'] + 1)
    return pc_probs, vals


def pc_a(a, params, model):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    pc_ab_probs, _ = pc_ab(a, b, params, model)
    pb_probs, _ = pb(params, model)
    pc_ab_probs = pc_ab_probs * pb_probs
    pc_probs = pc_ab_probs.sum(axis=2)
    vals = np.arange(params['amax'] + params['bmax'] + 1)
    return pc_probs, vals


def pc_b(b, params, model):
    a = np.arange(params['amin'], params['amax'] + 1)
    pc_ab_probs, _ = pc_ab(a, b, params, model)
    pa_probs, _ = pa(params, model)
    pc_ab_probs = pc_ab_probs * pa_probs.reshape(-1, 1)
    pc_probs = pc_ab_probs.sum(axis=1)
    vals = np.arange(params['amax'] + params['bmax'] + 1)
    return pc_probs, vals


def pd_c(c, params, model):
    dmax = 2 * (params['amax'] + params['bmax']) + 1
    vals = np.arange(dmax)
    c_probs, _ = pc(params, model)
    probs = binom.pmf(np.arange(dmax), c.reshape(-1, 1), params['p3'])
    rows, column_indices = np.ogrid[: probs.shape[0], : probs.shape[1]]
    column_indices = column_indices - c[:, np.newaxis]
    probs = probs[rows, column_indices]
    return probs.T, vals


def pd(params, model):
    c = np.arange(params['amax'] + params['bmax'] + 1)
    pd_c_probs, _ = pd_c(c, params, model)
    pc_probs, _ = pc(params, model)
    pd_c_probs = pd_c_probs * pc_probs.reshape(1, -1)
    pd_probs = pd_c_probs.sum(axis=1)
    vals = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    return pd_probs, vals


def pc_d(d, params, model):
    vals = np.arange(params['amax'] + params['bmax'] + 1)

    pd_probs, _ = pd(params, model)
    pd_probs = pd_probs[d]
    pd_c_probs, _ = pd_c(vals, params, model)
    pd_c_probs = pd_c_probs[d]
    pc_probs, _ = pc(params, model)

    pc_d_probs = pd_c_probs * pc_probs.reshape(1, -1) / pd_probs.reshape(-1, 1)
    return pc_d_probs.T, vals


def pc_abd(a, b, d, params, model):
    vals = np.arange(params['amax'] + params['bmax'] + 1)

    pa_probs, vals_a = pa(params, model)
    pa_probs = pa_probs[:a.shape[0]]
    pb_probs, vals_b = pb(params, model)
    pb_probs = pb_probs[b.shape[0]]
    pd_c_probs, _ = pd_c(vals, params, model)
    pd_c_probs = pd_c_probs[d]
    pc_ab_probs, _ = pc_ab(a, b, params, model)

    probs = np.ones((params['amax'] + params['bmax'] + 1, a.shape[0], b.shape[0], d.shape[0]))
    probs *= pc_ab_probs[..., np.newaxis] * np.expand_dims(pd_c_probs.T, axis=(1, 2))
    probs *= pa_probs.reshape(1, -1, 1, 1) * pb_probs.reshape(1, 1, -1, 1)
    probs /= probs.sum(axis=0)
    return probs, vals