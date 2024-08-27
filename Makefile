# This Makefile compiles Cython
.POSIX:

.PHONY: cython

all: cython

clean:
	python3 setup.py clean

cython:
	python3 setup.py build_ext -b src/
