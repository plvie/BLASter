# SeysenLLL

SeysenLLL: Lattice reduction à la LLL but using Seysen's reduction technique instead of size reduction.

## Requirements

- python3
- Cython (version 3.0 or later)
- Other python modules: numpy setuptools threadpoolctl (installed system-wide or in locally through `make venv`)
- libeigen3-dev (installed system-wide or locally through `make eigen3`)

Optional:

- Python module: virtualenv (for creating a local virtual environment to install python3 modules)
- fplll (for generating q-ary lattices with the `latticegen` command)

## Building

- Optional: Run `make eigen3` to install libeigen3 library in a subdirectory.
- Optional: Run `make venv` to create a local virtual environment and install the required python3 modules.
- Run `make` to compile all the Cython.

## Running

- Run the command by e.g. typing `src/app.py -pvi INPUTFILE`.

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
