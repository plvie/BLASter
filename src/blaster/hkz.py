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
from fpylll.util import gaussian_heuristic

import numpy as np
from fractions import Fraction
from math import gcd, ceil,floor
from functools import reduce

# def float64_to_integer_matrix(A: np.ndarray):
#     """
#     Convertit une matrice float64 en une matrice d'entiers exacts + facteur d'échelle.
#     Retourne (M_int, S) tels que A = M_int / S exactement.
#     """
#     # 1. convertir en rationnels
#     R = [[Fraction(x).limit_denominator() for x in row] for row in A]
#     # 2. lister tous les dénominateurs
#     dens = [f.denominator for row in R for f in row]
#     # 3. calculer le PPCM
#     def lcm(a, b): 
#         return a * b // gcd(a, b)
#     S = reduce(lcm, dens, 1)
#     # 4. construire la matrice d’entiers (dtype=object pour éviter overflow)
#     M_int = np.empty((A.shape[0], A.shape[1]), dtype=object)
#     for i, row in enumerate(R):
#         for j, f in enumerate(row):
#             M_int[i, j] = f.numerator * (S // f.denominator)
#     return M_int, S

# def to_fpylll_integer_matrix(M):
#     """
#     Convertit un np.ndarray dtype=object ou une liste de listes Python ints
#     en fpylll.IntegerMatrix.
#     """
#     if hasattr(M, 'shape'):
#         n, m = M.shape
#     else:
#         n = len(M)
#         m = len(M[0]) if n else 0

#     IM = IntegerMatrix(n, m)
#     for i in range(n):
#         for j in range(m):
#             # M peut être array dtype=object ou list
#             val = M[i][j] if getattr(M, 'dtype', None) == object else M[i, j]
#             IM[i, j] = val
#     return IM

def float64_to_integer_matrix(A):
    max_abs = np.nanmax(np.abs(A))
    # n = A.shape[0]
    # print(max_abs)
    # flat_index = np.nanargmax(absA)

    # # 3) convert flat index back to 2D indices
    # i, j = np.unravel_index(flat_index, A.shape)

    # print("max |A| =", max_abs, "at (i,j) =", (i, j))
    scale_factor = (2**62) // (int(max_abs) + 1) - 1
    return (A * scale_factor).astype(np.int64), scale_factor

def to_fpylll_integer_matrix(M):
    IM = IntegerMatrix.from_matrix(M.T)
    return IM

def hkz_kernel(A,n):
    # Pool.map only supports a single parameter
    if isinstance(A, IntegerMatrix):
        IM = A
    elif isinstance(A, np.ndarray):
        if A.dtype == np.float64:
            M_int, scale_factor = float64_to_integer_matrix(A)
            IM = to_fpylll_integer_matrix(M_int)
        else:
            raise TypeError(f"Unsupported NumPy dtype {A.dtype}")
    else:
        raise TypeError(f"Unsupported matrix type {type(A)}")
    params = {"pump__down_sieve": True, "pump__down_stop": 9999, "saturation_ratio":1, "pump__prefer_left_insert":10, "workout__dim4free_min":0,"workout__dim4free_dec":15}
    kwds_ = OrderedDict()
    for k, v in params.items():
        k_ = k.replace("__", "/")
        kwds_[k_] = v
    params = kwds_
    params = SieverParams(**params)
    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)
    gso = GSO.Mat(IM, U = IntegerMatrix.identity(IM.nrows), UinvT = IntegerMatrix.identity(IM.nrows))        # on construit l’objet GSO
    #gso.update_gso()

    g6k = Siever(gso, params)
    tracer = dummy_tracer
    # runs a workout woth pump-down down until the end
    workout(g6k, tracer, 0, n, pump_params=pump_params, **workout_params)
    #Just making sure
    # pump(g6k, tracer, 15, n-15, 0, **pump_params)
    g6k.lll(0, n)
    # g6k.update_gso(0,n)
    U = g6k.M.U
    U_np = np.empty((U.nrows, U.ncols), dtype=np.int64)
    U.to_matrix(U_np)
    #check that the B[0] is the smallest vector of the basis
    B = g6k.M.B
    B_np = np.empty((B.nrows, B.ncols), dtype=np.int64)
    B.to_matrix(B_np)

    # 2) compute squared‐lengths of each basis vector
    #    (we use squared‐length so there’s no slow sqrt)
    sq_norms = np.einsum('ij,ij->i', B_np, B_np)

    # 3) find the index of the shortest vector
    idx_shortest = int(np.argmin(sq_norms))
    print("index of shortest row:", idx_shortest)

    # # 4) assertion
    # assert idx_shortest == 0, f"shortest vector is at row {idx_shortest}, not 0!"
    return np.ascontiguousarray(U_np.T)

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

# def hkz():
#     """
#     Attempt HKZ reduction. 
#     """
#     description = hkz.__doc__

#     args, all_params = parse_args(description,
#                                   challenge_seed=0,
#                                   pump__down_sieve=True,
#                                   pump__down_stop=9999,
#                                   saturation_ratio=.8,
#                                   pump__prefer_left_insert=10,
#                                   workout__dim4free_min=0, 
#                                   workout__dim4free_dec=15
#                                   )

#     stats = run_all(hkz_kernel, list(all_params.values()),
#                     lower_bound=args.lower_bound,
#                     upper_bound=args.upper_bound,
#                     step_size=args.step_size,
#                     trials=args.trials,
#                     workers=args.workers,
#                     seed=args.seed
#                     )

#     inverse_all_params = OrderedDict([(v, k) for (k, v) in all_params.items()])

#     for (n, params) in stats:
#         stat = stats[(n, params)]
#         if stat[0] is None:
#             logging.info("Trace disabled")
#             continue

#         if len(stat) > 0:
#             cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
#             walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
#             flast = sum([float(node["flast"]) for node in stat])/len(stat)
#             avr_db, max_db = db_stats(stat)
#             fmt = "%48s :: m: %1d, n: %2d, cputime :%7.4fs, walltime :%7.4fs, flast : %2.2f, avr_max db: 2^%2.2f, max_max db: 2^%2.2f" # noqa
#             logging.info(fmt % (inverse_all_params[params], params.threads, n, cputime, walltime, flast, avr_db, max_db))
#         else:
#             logging.info("Trace disabled")

#     if args.pickle:
#         pickler.dump(stats, open("hkz-asvp-%d-%d-%d-%d.sobj" %
#                                  (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))



# if __name__ == '__main__':
#     hkz()/home/paul/LWE_attack/hkz.py