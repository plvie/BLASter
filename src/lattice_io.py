"""
Read/write a matrix in FPLLL's latticegen format, into/from numpy array.
"""
import numpy as np


def read_matrix(input_file, verbose=False, reverse=True):
    """
    Read a matrix from a file, or from stdin.
    :param input_file: file name, or None (reads from stdin).
    :param verbose: ask for input when having no file name
    :return: a matrix
    """
    data = []
    if input_file is None:
        if verbose:
            print("Supply a matrix in fpLLL format:")
        data.append(input())
        while data[-1][-2] != ']':
            data.append(input())
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data.append(f.readline()[:-1])
            while data[-1][-2] != ']':
                data.append(f.readline()[:-1])

    # Strip away starting '[' and ending ']'
    data[0] = data[0][1:]
    data[-1] = data[-1][:-1]

    # Convert data to list of integers
    for i in range(len(data)):
        data[i] = list(map(int, data[i][1:-1].split(' ')))

    if reverse:
        data.reverse()

    # Use column vectors.
    return np.array(data, dtype=np.int64).transpose()


def write_matrix(output_file, basis, reverse=True):
    # Assume that the basis is given with column vectors as input.
    # However, output them as row vectors.
    basis = basis.transpose()

    if reverse:
        basis = list(reversed(basis))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[')
        for (i, v) in enumerate(basis):
            f.write('[' + ' '.join(map(str, v)) + (']\n' if i < len(basis) - 1 else ']]\n'))
