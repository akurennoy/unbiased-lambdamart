#!/bin/sh

gcc -c -fPIC argsort.cpp -o argsort.o
gcc argsort.o -shared -o libargsort.so
rm argsort.o
python setup_linux.sh build_ext --inplace
