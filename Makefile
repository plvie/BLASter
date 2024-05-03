# This Makefile compiles the C-parts of the python program dual_utils.py
.POSIX:

all: lll.so

clean:
	rm -f lll.so

CXX=g++
CXXFLAGS=-Wall --std=c++17

lll.so: lll.cpp
	$(CXX) $(CXXFLAGS) -fPIC -Ofast -march=native -shared -o lll.so lll.cpp $(LIBS)
# $(CXX) $(CXXFLAGS) -fPIC -g -fsanitize=address,undefined -shared -o lll.so lll.cpp $(LIBS)

