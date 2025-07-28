from pathlib import Path
from sys import argv, platform
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

import numpy as np


# Make sure OpenMP is used in Cython and Eigen.
openmp_arg = '/openmp' if platform.startswith("win") else '-fopenmp'
include_dirs = [np.get_include()]

# Look for the Eigen library in `/usr/include` and `~/.local/include`.
for f in [Path('eigen3'), Path('/usr/include/eigen3'), Path.home().joinpath('.local/include/eigen3')]:
    if f.exists() and f.is_dir():
        include_dirs += [str(f)]
        break
else:
    print("ERROR: Eigen3 library is required!")
    print("NOTE : Please run 'make eigen3'")
    exit(1)

# Compile with extra arguments
compile_args = [
    '--std=c++17',
    '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION',
    openmp_arg,
]

# Link with extra arguments
link_args = [
    openmp_arg,
    '-Ofast',
    '-march=native',
    '-mtune=native',
    '-flto=6',
    '-ffast-math',
    '-ffp-contract=fast',
    '-fno-math-errno',
    '-funroll-loops',
    '-fpeel-loops',
    '-ftree-vectorize',
    '-fivopts',
    '-fprefetch-loop-arrays',
    '-falign-loops=64',
    '-fopenmp',
    '-fopenmp-simd',
    '-fstrict-aliasing',
    '-fno-exceptions',
    '-fno-rtti',
    '-fvisibility=hidden',
    '-fno-trapping-math',
    '-DEIGEN_NO_DEBUG',
    '-DNDEBUG',
    '-Wl,--gc-sections'
]


if '--cython-gdb' in argv:
    # Debug arguments
    debug_args = [
        '-O1',
        '-fsanitize=address,undefined',
        '-g',
        '-fno-omit-frame-pointer',
    ]
    compile_args += debug_args
    link_args += debug_args
else:
    # "Release" arguments
    compile_args += [
    '-Ofast',
    '-march=native',
    '-mtune=native',
    '-flto=6',
    '-ffast-math',
    '-ffp-contract=fast',
    '-fno-math-errno',
    '-funroll-loops',
    '-fpeel-loops',
    '-ftree-vectorize',
    '-fivopts',
    '-fprefetch-loop-arrays',
    '-falign-loops=64',
    '-fopenmp',
    '-fopenmp-simd',
    '-fstrict-aliasing',
    '-fno-exceptions',
    '-fno-rtti',
    '-fvisibility=hidden',
    '-fno-trapping-math',
    '-DEIGEN_NO_DEBUG',
    '-DNDEBUG',
    '-Wl,--icf=all',
    '-Wl,--gc-sections'
    ]

extensions = [Extension(
    name="blaster_core",
    sources=["core/blaster.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args
)]

setup(
    name="blaster",                        # <— the distribution & import name
    version="0.1.0",
    packages=find_packages(where="src"),   # <— finds src/blaster/
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, language_level="3", build_dir='build/cpp'),
    options={'build': {'build_lib': 'src/'}},
)
