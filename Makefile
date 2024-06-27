# This Makefile compiles Cython
.POSIX:

.PHONY: cython

all: cython

clean:
	python setup.py clean
	rm src/seysen_lll.cpp src/seysen_lll.html

cython:
	python setup.py build_ext --inplace
