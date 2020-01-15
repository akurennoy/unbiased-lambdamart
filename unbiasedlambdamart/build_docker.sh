#! /bin/bash
/opt/python/cp36-cp36m/bin/pip3.6 install cython
make libargsort.a 
/opt/python/cp36-cp36m/bin/python3.6 setup_linux.py build_ext --inplace
make clean