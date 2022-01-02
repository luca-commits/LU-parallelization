include ../../../config.mk

EXTRA_FLAGS=-DPOLYBENCH_TIME -Wall
CFLAGS=-O3 -I"${MKLROOT}/include" -std=c99 
MKL_LINKING_FLAGS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

CC=icc
MPICC=mpiicc

all: lu-mpi lu-scalapack lu-omp

lu-mpi: lu-mpi.c lu.h
	${VERBOSE} ${MPICC} -o ./bin/lu-mpi lu-mpi.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

lu-scalapack: lu-scalapack.c lu.h
	${VERBOSE} ${MPICC} -o ./bin/lu-scalapack lu-scalapack.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

lu-omp: lu-omp.c lu.h
	${VERBOSE} ${CC} -qopenmp -o ./bin/lu-omp lu-omp.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}