from collections import namedtuple

import numpy as np

from lattice_io import read_qary_lattice
from seysen_lll import svp_enumerate
from seysen import seysen_lll
from stats import gaussian_heuristic


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000, threshold=2147483647, suppress=True)
    L = read_qary_lattice('input/128')

    Args = namedtuple('Args', 'logfile delta depth cores verbose LLL')
    args = Args(logfile=None, delta=0.99, depth=1, cores=1, verbose=False, LLL=64)
    U, B_red, tprof = seysen_lll(L, args)

    B_sub = B_red[:, :16]

    v0 = B_sub[:, 1]
    print(f'Currently shortest vector has length {np.dot(v0, v0)**.5:.3f}')

    R = np.linalg.qr(B_sub, mode='r')
    assert R.shape == (16, 16)

    sol = svp_enumerate(R, np.repeat(R[0][0]*R[0][0], 16))
    print("Solution: ", sol)
    v = B_sub @ sol
    print(f'Vector: {v} has length {np.dot(v, v)**.5:.3f}')
    print(f'Gaussian Heuristic expects length {gaussian_heuristic(B_sub):.3f}')
