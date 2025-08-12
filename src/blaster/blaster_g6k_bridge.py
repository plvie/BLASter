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

from g6k.algorithms.pump import pump
from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.siever_params import SieverParams
import six
from fpylll import IntegerMatrix, GSO
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

import numpy as np
from math import log, sqrt

import gc
import cupy as cp
import time
import sys

from sage.all import Matrix as SageMatrix, ZZ

#RHF = 1.01267^n, slope = -0.039489, ∥b_1∥ = 631.0 Total time: 199.026s


scale_factor = 2**52

to_pyint = np.vectorize(int, otypes=[object])

def svp_kernel_solver(g6k, eta, target_norm,kappa, workout_params, pump_params=None):
    """
    Solve SVP using workout with saturation control up to target_norm^2.

    Args:
        B: basis matrix
        params: dict of G6K parameters
        eta: dimension reduction parameter for sub-block
        target_norm: squared norm target to reach
        workout_params: dict of workout-specific parameters
        pump_params: dict of pump-specific parameters (optional fallback)
    Returns:
        g6k Siever instance with reduced basis reaching target_norm
    """
    # Setup tracer (dummy by default)
    tracer = dummy_tracer
    d = g6k.full_n
    # Quick check: if already short enough
    
    f = max(0, d - eta - kappa)
    print(f"Starting sieving on block [ {kappa}, {d} ) with a sieve of dim {eta}, so on [{f+kappa}, {d})")

    #add some params
    # params = {, "threads": 16}#, "reserved_n": beta, "db_size_factor": 4}
    # kwds_ = OrderedDict()
    # for k, v in params.items():
    #     k_ = k.replace("__", "/")
    #     kwds_[k_] = v
    # params = kwds_
    # params = SieverParams(**params)
    # g6k.params = params

    pump(g6k, tracer, kappa, d-kappa, f, verbose=True, down_sieve=True, goal_r0=((target_norm*scale_factor) **2) ) #already projected


def float64_to_integer_matrix(A):
    global scale_factor

    _, exponents = np.frexp(A)
    min_exponent = max(0, int(exponents.min())) # not use a too high scale factor if it's already high
    scale_factor = 2**(52-min_exponent)
    A_scaled = ((A.astype(np.float128) * scale_factor)) #float128 for safety compute
    y_int = to_pyint(A_scaled)

    #if debug
    recon = np.array(y_int, dtype=np.float128) / scale_factor
    if not np.allclose(recon, A.astype(np.float128), atol=np.finfo(np.float64).eps): # because BLAS (QR before) is already at some machine epsilon float64
        errors = np.abs(A.astype(np.float128) - recon)
        max_error = errors.max()
        print("warning : loss precision detected")
        print("max error", max_error)
    
    return y_int # more precise conversion

#build it with -y and don't remove threads params
def g6k_kernel(A,n, beta, jump, target_norm=None, kappa=0):
    if isinstance(A, IntegerMatrix):
        IM = A
    elif isinstance(A, np.ndarray):
        if A.dtype == np.float64:
            max_abs = np.nanmax(np.abs(A))
            # scale_factor = (2**62) // (int(max_abs) + 1) - 1
            # print(np.nanmax(np.abs(A - float64_to_integer_matrix(A).astype(np.float64)/scale_factor)))
            IM = IntegerMatrix.from_matrix(float64_to_integer_matrix(A).T)
        else:
            IM = IntegerMatrix.from_matrix(A.T)
    else:
        raise TypeError(f"Unsupported matrix type {type(A)}")
    params = {"pump__down_sieve": True, "pump__down_stop":15, "threads": 16}#, "reserved_n": beta, "db_size_factor": 4}
    #1.33 faster in dim 100 according to g6k paper
    kwds_ = OrderedDict()
    for k, v in params.items():
        k_ = k.replace("__", "/")
        kwds_[k_] = v
    params = kwds_
    params = SieverParams(**params)
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)

    # gso = GSO.Mat(IM, U = IntegerMatrix.identity(IM.nrows), UinvT = IntegerMatrix.identity(IM.nrows)
    # , float_type="long double", flags=GSO.ROW_EXPO)
    g6k = Siever(IM, params)
    tracer = dummy_tracer
    #other mode possible 
    # workout(g6k, tracer, 0, n, pump_params=pump_params, **workout_params)
    if target_norm:
        svp_kernel_solver(g6k, beta, target_norm, kappa, workout_params, pump_params)
        #pump(g6k, tracer, 0, n, 0, **pump_params, verbose=True, goal_r0=proj_target_norm)
    else:
        if n <= beta:
                dim4free = default_dim4free_fun(n)
                pump(g6k, tracer, 0, n, dim4free, **pump_params, verbose=True)
        else:
            i = 0
            end = n - beta
            while i < end:
                length = min(jump, end - i)
                dim4free = default_dim4free_fun(beta + length - 1)
                pump(g6k, tracer, i, beta + length - 1, dim4free, **pump_params, verbose=True)
                i += length
    B = g6k.M.B
    if A.dtype == np.float64:
        A_np = float64_to_integer_matrix(A).T
    else:
        A_np = A.T
    B_np = np.empty((B.nrows, B.ncols), dtype=object)
    B.to_matrix(B_np)
    # need to use sage sometimes because solve is only float64, so if you need more precision do in sage (like now with the better conversion)
    # more precise for a really small cost
    A_list = A_np.tolist()
    B_list = B_np.tolist()
    A_sage = SageMatrix(ZZ, A_list)
    B_sage = SageMatrix(ZZ, B_list)
    #U = np.rint(np.linalg.solve(A_np.T,B_np.T)).astype(np.int64)
    # 4) Solve A^T * U = B^T exactly: U_sage = (A_sage^T)^{-1} * B_sage^T
    U_sage = A_sage.transpose().solve_right(B_sage.transpose())

    # 5) Convert back to numpy int64
    U = U_sage.numpy()

    # 6) Verification
    # assert np.array_equal(A_np.T @ U, B_np.T)

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