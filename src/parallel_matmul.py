import numpy as np
import threading
from threadpoolctl import threadpool_limits


# Based on:
# https://stackoverflow.com/questions/35101312/multi-threaded-integer-matrix-multiplication-in-numpy-scipy
def _contiguous_dot(a, b, out):
    with threadpool_limits(limits=1):
        np.dot(a, b, out)


def pardot(a, b, num_cores):
    """
    Return the matrix product a * b.
    The product is split into `num_cores` row partitions that are performed in parallel threads.
    """
    assert a.shape[1] == b.shape[0], "Dimension mismatch"
    assert a.dtype == b.dtype, "Type mismatch"

    if num_cores == 1:
        with threadpool_limits(limits=1):
            return a @ b

    num_rows, num_cols = a.shape[0], b.shape[1]
    step = num_rows // num_cores
    out = np.empty((num_rows, num_cols), dtype=a.dtype)
    threads = []
    for i in range(step, num_rows, step):
        j = min(i + step, num_rows)
        th = threading.Thread(target=_contiguous_dot, args=(a[i:j, :], b, out[i:j, :]))
        th.start()
        threads.append(th)

    # run it for i=0, j=step
    _contiguous_dot(a[0:step, :], b, out[0:step, :])

    for th in threads:
        th.join()
    return out
