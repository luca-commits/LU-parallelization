include ../../../config.mk

EXTRA_FLAGS= -DSMALL_DATASET -DPOLYBENCH_TIME -Wall 
CFLAGS=-O3 -I"${MKLROOT}/include" -std=c99 
MKL_LINKING_FLAGS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
CC=mpiicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -fopenmp -o lu lu.c ${CFLAGS} -D SMALL_DATASET -DPOLYBENCH_TIME -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

#lu_med: lu.c lu.h
#	${VERBOSE} ${CC} -fopenmp -o lu_med lu.c ${CFLAGS} -D MEDIUM_DATASET -DPOLYBENCH_TIME -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

#lu_lar: lu.c lu.h
#	${VERBOSE} ${CC} -fopenmp -o lu_lar lu.c ${CFLAGS} -D LARGE_DATASET -DPOLYBENCH_TIME -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

#lu_elar: lu.c lu.h
#	${VERBOSE} ${CC} -fopenmp -o lu_elar lu.c ${CFLAGS} -D EXTRALARGE_DATASET -DPOLYBENCH_TIME -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f lu

