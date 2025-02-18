# BLASter

BLASter is a proof of concept of an LLL-like lattice reduction algorithm that uses:

- parallelization,
- segmentation,
- Seysen's reduction instead of size reduction, and
- a linear algebra library.

## Disclaimer

The goal of this software is to showcase speed ups that are possible in lattice reduction software.
This software is a *proof of concept*!

In particular, we **do not**:

- guarantee the algorithm terminates, nor claim its output is correct on all lattices,
- support lattices with large entries,
- consider issues / PRs that improve efficiency or robustness,
- actively maintain this software.

We **do**:

- happily answer any questions to explain design choices phrased as: *"Why is X done in Y way?"*. The answer may, in many cases, be: "because it is faster in practice".
- encourage the cryptographic community to build a new robust lattice reduction library incorporating the ideas in this proof of concept.

## Requirements

- python3
- Cython version 3.0 or later
- Python modules: `cysignals numpy setuptools` (installed system-wide or in locally through `make venv`)
- The Eigen library version 3 or later (installed system-wide or locally through `make eigen3`)

Optional:

- Python module: virtualenv (for creating a local virtual environment to install python3 modules)
- fplll (for generating q-ary lattices with the `latticegen` command)

## Building

- Optional: Run `make eigen3` to install libeigen3 library in a subdirectory.
- Optional: Run `make venv` to create a local virtual environment and install the required python3 modules.
- Run `make` to compile all the Cython files in `core/`.

## Debugging

- Debug the C++/Cython code with the `libasan` and `libubsan` sanitizers by running `make cython-gdb`.
    These sanitizers check for memory leaks, out of bounds accesses, and undefined behaviour.
- When executing the `src/app.py`, preload libasan like this:
    `LD_PRELOAD=$(gcc -print-file-name=libasan.so) src/app.py -pvi INPUTFILE`
- If you want to run the program with the `gdb` debugger, read the [Cython documentation](https://cython.readthedocs.io/en/stable/src/userguide/debugging.html#running-the-debugger), for more info.

## Running

*Note: you first need to build the software, see above.*

Run the command by e.g. typing `./python3 src/app.py -pvi INPUTFILE`.
Add `-h` for seeing all available command line arguments.

## Example

Command: `time latticegen q 128 64 20 p | src/app.py -pq`.

Expected output:
```
Profile: [9.38 9.39 9.30 9.28 9.18 ... 4.27 4.33 4.31 4.29 4.35]
Root Hermite factor: 1.020447, ∥b_1∥ = 11906.636

real	0m0,754s
user	0m2,271s
sys	0m0,105s
```

To run deep-LLL with depth `4`, run `src/app.py -pq -i {lattice} -d4`.

To run progressive BKZ-60 (with 4-deep-LLL) with `1` tours and block size increments of `2`, run `src/app.py -pq -i {lattice} -b60 -t1 -P2`.
