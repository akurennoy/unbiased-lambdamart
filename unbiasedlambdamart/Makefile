CXX ?= g++-8

libargsort.a: argsort.o
	ar rcs libargsort.a argsort.o

argsort.o: argsort.cpp argsort.h
	${CXX} ${CXXFLAGS} -c -fPIC argsort.cpp

clean:
	rm -f libargsort.a argsort.o
