from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


exts = [
    Extension(
        name="lambdaobj",
        sources=["lambdaobj.pyx"],
        libraries=["argsort"],
        library_dirs=["."],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++"
    )
]

setup(ext_modules=cythonize(exts))
