#!/usr/bin/env python
# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####


"""
Full Sieve Command Line Client
"""

from __future__ import absolute_import
import pickle as pickler
import logging
from collections import OrderedDict

from g6k.algorithms.workout import workout, pump
from g6k.siever import Siever
from g6k.siever_params import SieverParams
import six
from fpylll import IntegerMatrix, GSO
from fpylll.tools.bkz_stats import dummy_tracer

import numpy as np

import gc
import cupy as cp

#RHF = 1.01267^n, slope = -0.039489, ∥b_1∥ = 631.0 Total time: 199.026s


def float64_to_integer_matrix(A):
    max_abs = np.nanmax(np.abs(A))
    scale_factor = (2**62) // (int(max_abs) + 1) - 1
    return (A * scale_factor).astype(np.int64)

def hkz_kernel(A,n, beta, pump_and_jump):
    if isinstance(A, IntegerMatrix):
        IM = A
    elif isinstance(A, np.ndarray):
        if A.dtype == np.float64:
            IM = IntegerMatrix.from_matrix(float64_to_integer_matrix(A).T)
        else:
            raise TypeError(f"Unsupported NumPy dtype {A.dtype}")
    else:
        raise TypeError(f"Unsupported matrix type {type(A)}")
    params = {"pump__down_sieve": True, "workout__dim4free_min":0,"workout__dim4free_dec":1}
    kwds_ = OrderedDict()
    for k, v in params.items():
        k_ = k.replace("__", "/")
        kwds_[k_] = v
    params = kwds_
    params = SieverParams(**params)
    params = params.new(#db_size_base   = 1.1,    # ou 1.25, 1.333… selon ce que tu veux
    # db_size_factor = 3.5,      # augmentation exponentielle
    # db_limit       = 10_000_000,  # plafonné à 10 M
    # sample_by_sums = True,
    # saturation_ratio  = 0.8)
    )
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)

    gso = GSO.Mat(IM, U = IntegerMatrix.identity(IM.nrows), UinvT = IntegerMatrix.identity(IM.nrows)
    , float_type="long double", flags=GSO.ROW_EXPO)
    g6k = Siever(gso, params)
    tracer = dummy_tracer
    #other mode possible 
    # workout(g6k, tracer, 0, n, pump_params=pump_params, **workout_params)
    if pump_and_jump:
        pump_n_jump_bkz_tour(g6k, tracer, beta, pump_params=pump_params)
    else:
        jump = 1
        if n <= beta:
            pump(g6k, tracer, 0, n, 0, **pump_params)
        else:
            for i in range(0,n-beta+1, jump):
                pump(g6k, tracer, i, beta, 0, **pump_params)
    B = g6k.M.B
    A_np = (float64_to_integer_matrix(A).T)
    B_np = np.empty((B.nrows, B.ncols), dtype=int)
    B.to_matrix(B_np)
    U = np.rint(np.linalg.solve(A_np.T,B_np.T)).astype(np.int64)

    # Cleanup before return just to be extra safe about memory usage
    del IM, gso, g6k, B, B_np, A_np, params, pump_params, workout_params, tracer

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    # assert np.allclose(np.dot(A_np.T, U), B_np.T)
    return U

def pop_prefixed_params(prefix, params):
    """
    pop all parameters from ``params`` with a prefix.

    A prefix is any string before the first "/" in a string::

        >>> pop_prefixed_params('foo', {'foo/bar': 1, 'whoosh': 2})
        {'bar': 1}

    :param prefix: prefix string
    :param params: key-value store where keys are strings

    """
    keys = [k for k in params]
    poped_params = {}
    if not prefix.endswith("/"):
        prefix += "/"

    for key in keys:
        if key.startswith(prefix):
            poped_key = key[len(prefix):]
            poped_params[poped_key] = params.pop(key)

    return poped_params


def dim4free_wrapper(dim4free_fun, blocksize):
    """
    Deals with correct dim4free choices for edge cases when non default
    function is chosen.

    :param dim4free_fun: the function for choosing the amount of dim4free
    :param blocksize: the BKZ blocksize

    """
    if blocksize < 40:
        return 0
    dim4free = dim4free_fun(blocksize)
    return int(min((blocksize - 40)/2, dim4free))

def default_dim4free_fun(blocksize):
    """
    Return expected number of dimensions for free, from exact-SVP experiments.

    :param blocksize: the BKZ blocksize

    """
    return int(11.5 + 0.075*blocksize)


def pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=1,
                         dim4free_fun=default_dim4free_fun, extra_dim4free=0,
                         pump_params=None, goal_r0=0., verbose=False):
    """
    Run a PumpNjump BKZ-tour: call Pump consecutively on every (jth) block.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param blocksize: dimension of the blocks
    :param jump: only call the pump every j blocks
    :param dim4free_fun: number of dimension for free as a function of beta (function, or string
        e.g. `lambda x: 11.5+0.075*x`)
    :param extra_dim4free: increase the number of dims 4 free (blocksize is increased, but not sieve
        dimension)
    :param pump_params: parameters to pass to the pump
    """
    if pump_params is None:
        pump_params = {"down_sieve": False}

    if "dim4free" in pump_params:
        raise ValueError("In pump_n_jump_bkz, you should choose dim4free via dim4free_fun.")

    d = g6k.full_n
    g6k.shrink_db(0)
    g6k.lll(0,d)
    g6k.update_gso(0,d)

    if isinstance(dim4free_fun, six.string_types):
        dim4free_fun = eval(dim4free_fun)

    dim4free = dim4free_wrapper(dim4free_fun, blocksize) + extra_dim4free
    blocksize += extra_dim4free

    indices  = [(0, blocksize - dim4free + i, i) for i in range(0, dim4free, jump)]
    indices += [(i, blocksize, dim4free) for i in range(0, d - blocksize, jump)]
    indices += [(d - blocksize + i, blocksize - i, dim4free - i) for i in range(0, dim4free, jump)]

    pump_params["down_stop"] = dim4free+3

    for (kappa, beta, f) in indices:
        if verbose:
            print( "\r k:%d, b:%d, f:%d " % (kappa, beta, f), end='')
            sys.stdout.flush()

        pump(g6k, tracer, kappa, beta, f, **pump_params)
        g6k.lll(0, d)
        if g6k.M.get_r(0, 0) <= goal_r0:
            return

    if verbose:
        print( "\r k:%d, b:%d, f:%d " % (d-(blocksize-dim4free), blocksize-dim4free, 0), end='')
        sys.stdout.flush()

    pump_params["down_stop"] = blocksize - dim4free
    pump(g6k, tracer, d-(blocksize-dim4free), blocksize-dim4free, 0, **pump_params)
    if verbose:
        print('')
        sys.stdout.flush()