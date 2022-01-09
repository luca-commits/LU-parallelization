include ../../config.mk

EXTRA_FLAGS=-Wall -check-pointers=rw #-march=skylake-avx512
CFLAGS=-std=c99 -O3
MKL_LINKING_FLAGS=-liomp5 -lpthread -lm -ldl

CC=icc
MPICC=mpiicc

all: ./bin/jacobi-2d-mpi ./bin/jacobi-2d-hybrid ./bin/jacobi-2d-omp

./bin/jacobi-2d-mpi: jacobi-2d-mpi.c jacobi-2d.h
	${VERBOSE} ${MPICC} -o ./bin/jacobi-2d-mpi jacobi-2d-mpi.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

./bin/jacobi-2d-hybrid: jacobi-2d-mpi.c jacobi-2d.h
	${VERBOSE} ${MPICC} -fopenmp -o ./bin/jacobi-2d-hybrid jacobi-2d-mpi.c -DHYBRID ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

./bin/jacobi-2d-omp: jacobi-2d-omp.c jacobi-2d.h
	${VERBOSE} ${CC} -fopenmp -o ./bin/jacobi-2d-omp jacobi-2d-omp.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}
