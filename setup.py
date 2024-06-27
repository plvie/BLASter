from pathlib import Path
from sys import platform
from setuptools import Extension, setup
from Cython.Build import cythonize



if platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

include_dirs = []
for f in [Path('/usr/include/eigen3'), Path.home().joinpath('.local/include/eigen3')]:
    if f.exists() and f.is_dir():
        include_dirs += [str(f)]
if not include_dirs:
    print("ERROR: seysen_lll requires the Eigen3 library!")
    exit(1)


extensions = [Extension(
    name="seysen_lll",
    sources=["src/seysen_lll.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=['--std=c++17', '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION', openmp_arg],
    extra_link_args=[openmp_arg],
)]

setup(
    ext_modules=cythonize(extensions, language_level="3", annotate=True)
)
