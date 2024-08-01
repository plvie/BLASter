# This Makefile compiles Cython
.POSIX:

.PHONY: cython

all: cython

clean:
	python3 setup.py clean
	rm src/seysen_lll.cpp src/seysen_lll.html

cython:
	python3 setup.py build_ext --inplace
