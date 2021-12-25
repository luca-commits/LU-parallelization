include ../../../config.mk

EXTRA_FLAGS= -DSMALL_DATASET -DPOLYBENCH_TIME -Wall 
CFLAGS=-O3 -I"${MKLROOT}/include" -DMKL_ILP64 -std=c99 
MKL_LINKING_FLAGS=${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
CC=mpiicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

scalapack_test: scalapack_test.c
	${VERBOSE} ${CC} -o scalapack_test scalapack_test.c ${CFLAGS} $(MKL_LINKING_FLAGS) ${EXTRA_FLAGS}


clean:
	@ rm -f lu

