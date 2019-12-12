from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext


exts = [Extension(name="lambdaobj",
                  sources=["lambdaobj.pyx"],
                  libraries=["argsort"],
                  library_dirs=["."],
                  extra_compile_args=["-fopenmp"]
                  )]

setup(ext_modules=cythonize(exts))
