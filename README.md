# SeysenLLL

SeysenLLL: Lattice reduction à la LLL but using Seysen's reduction technique instead of size reduction.

## Requirements

- python3
- libeigen3-dev

Optional:

- fplll (for generating q-ary lattices with the `latticegen` command)

## Running

- Run `make` to compile all the Cython.
- Run the command by e.g. typing `src/app.py -pvi INPUTFILE`.


## Example

Command: `time latticegen q 128 64 20 p | src/app.py -pq`.

Expected output:
```
Profile: [9.38 9.39 9.30 9.28 9.18 9.30 9.28 9.22 9.22 9.11 9.08 9.00 9.02 8.98 8.93 8.92 8.87 8.81 8.74 8.62 8.71 8.63 8.60 8.51 8.40 8.43 8.38 8.31 8.32 8.32 8.25 8.27 8.20 8.07 8.09 8.06 8.12 8.19 8.11 8.01 7.95 7.85 7.87 7.78 7.72 7.58 7.55 7.50 7.43 7.52 7.51 7.42 7.35 7.26 7.20 7.16 7.19 7.13 7.13 7.01 6.93 6.86 6.94 6.84 6.83 6.85 6.76 6.69 6.67 6.67 6.57 6.50 6.39 6.27 6.27 6.23 6.17 6.19 6.15 6.21 6.12 6.06 6.11 6.00 5.95 5.91 5.93 5.88 5.82 5.83 5.73 5.66 5.56 5.49 5.41 5.40 5.33 5.34 5.32 5.33 5.26 5.17 5.08 5.07 5.08 5.16 5.04 4.98 4.85 4.78 4.82 4.75 4.68 4.62 4.56 4.51 4.42 4.50 4.45 4.33 4.27 4.26 4.15 4.27 4.33 4.31 4.29 4.35]
Root Hermite factor: 1.020447, ∥b_1∥ = 11906.636

real	0m0,754s
user	0m2,271s
sys	0m0,105s
```
