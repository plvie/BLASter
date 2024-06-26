# This Makefile compiles Cython
.POSIX:

.PHONY: cython

all: cython

clean:
	python setup.py clean

cython:
	python setup.py build_ext --inplace
