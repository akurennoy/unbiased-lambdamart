#!/bin/sh

gcc -c argsort.cpp
ar rcs libargsort.a argsort.o 
rm argsort.o
python setup_linux.sh build_ext --inplace
rm lambdaobj.c libargsort.a
rm -rf build
