"""
Read/write a matrix in fplll format, to/from numpy array.
"""
import numpy as np


def read_matrix(input_file, verbose=False, reverse=True):
    """
    Read a matrix from a file, or from stdin.
    :param input_file: file name, or when None, read from stdin.
    :param verbose: ask for input when having no file name
    :param reverse: whether to reverse the ordering of the outputted column vectors.
    :return: a matrix consisting of column vectors.
    """
    data = []
    if input_file is None:
        data.append(input("Supply a matrix in fplll format:\n" if verbose else ""))
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
    data = [list(map(int, line[1:-1].split(' '))) for line in data]

    if reverse:
        data.reverse()
    # Use column vectors.
    return np.ascontiguousarray(np.array(data, dtype=np.int64).transpose())


def write_matrix(output_file, basis):
    """
    Outputs a basis with column vectors to a file in fplll format.
    :param input_file: file name, or None (reads from stdin).
    :param basis: the matrix to output
    :param reverse: whether to reverse the ordering of the outputted column vectors.
    """
    basis = basis.transpose()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[')
        for (i, v) in enumerate(basis):
            f.write('[' + ' '.join(map(str, v)) + (']\n' if i < len(basis) - 1 else ']]\n'))
