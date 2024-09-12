from math import exp, gamma, log2, pi
import numpy as np


def get_profile(B, is_upper=False):
    """
    Return the profile of a basis, i.e. log_2 ||b_i*|| for i=1, ..., n.
    Note: the algorithm is taken base 2, just like https://github.com/keeganryan/flatter.
    :param B: basis for a lattice
    :param is_upper: whether B is already an upper triangular matrix or not
    """
    R = B if is_upper else np.linalg.qr(B, mode='r')
    return [log2(abs(d_i)) for d_i in R.diagonal()]


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


def gaussian_heuristic(B):
    """
    Return the Gaussian Heuristic for a particular basis.
    """
    rank = B.shape[1]
    return gh(rank) * 2.0**(sum(get_profile(B)) / rank)


def rhf(profile):
    """
    Return the root Hermite factor, given the profile of some basis, i.e.
        rhf(B) = (||b_0|| / det(B)^{1/n})^{1/(n-1)}.
    :param profile: profile belonging to some basis of some lattice
    """
    n = len(profile)
    return 2.0**((profile[0] - sum(profile) / n) / (n - 1))


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


def potential(profile):
    """
    Return the (log2 of the) potential of a basis profile.
    Normally in lattice reduction, this is a strictly decreasing function of time, and is used to
    prove that LLL runs in polynomial time.
    """
    n = len(profile)
    return sum((n - i) * profile[i] for i in range(n))
