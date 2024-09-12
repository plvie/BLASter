from math import exp, gamma, log, pi
import numpy as np


def get_profile(B):
    """
    Return the profile of a basis, i.e. log ||b_i*|| for i=1, ..., n.
    :param B: basis for a lattice
    """
    return [log(abs(d_i)) for d_i in np.linalg.qr(B, mode='r').diagonal()]


def gh(dim):
    """
    Return the Gaussian Heuristic at dimension n. This gives a prediction of
    the length of the shortest vector in a lattice of unit volume.
    :param n: lattice dimension
    :return: GH(n)
    """
    if dim >= 100:
        return float(dim / (2*pi*exp(1)))**0.5
    return float(gamma(1.0 + 0.5 * dim)**(1.0 / dim) / pi**0.5)


def rhf(profile):
    """
    Return the root Hermite factor, given the profile of some basis, i.e.
        rhf(B) = (||b_0|| / det(B)^{1/n})^{1/(n-1)}.
    :param profile: profile belonging to some basis of some lattice
    """
    n = len(profile)
    return exp((profile[0] - sum(profile) / n) / (n - 1))


def slope(profile):
    """
    Return the current slope of a profile
    """
    n = len(profile)
    i_mean = (n - 1) * 0.5
    x_mean = sum(profile)/n
    v1, v2 = 0.0, 0.0

    for i in range(n):
        v1 += (i - i_mean) * (profile[i] - x_mean)
        v2 += (i - i_mean) * (i - i_mean)
    return v1 / v2
