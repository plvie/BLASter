from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [Extension(
    name="seysen_lll",
    sources=["src/seysen_lll.pyx"],
    extra_compile_args=['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
)]

setup(
    ext_modules=cythonize(extensions, language_level="3")
)
