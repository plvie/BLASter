from pathlib import Path
from sys import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


# Make sure OpenMP is used in Cython and Eigen.
if platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

include_dirs = [np.get_include()]

# Look for the Eigen library in `/usr/include` and `~/.local/include`.
for f in [Path('/usr/include/eigen3'), Path.home().joinpath('.local/include/eigen3')]:
    if f.exists() and f.is_dir():
        include_dirs += [str(f)]
        break
else:
    print("ERROR: seysen_lll requires the Eigen3 library!")
    exit(1)

# Compile with extra arguments
compile_args = [
    '--std=c++17',
    '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION',
    '-DEIGEN_NO_DEBUG',
    openmp_arg,
]

extensions = [Extension(
    name="seysen_lll",
    sources=["core/seysen_lll.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=[openmp_arg],
)]

setup(
    ext_modules=cythonize(extensions, language_level="3", build_dir='build/cpp'),
    options={'build': {'build_lib': 'src/'}},
)
