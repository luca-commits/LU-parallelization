include ../../../config.mk

EXTRA_FLAGS= -DMINI_DATASET -DPOLYBENCH_TIME -Wall 
CFLAGS=-O3 -I"${MKLROOT}/include" -std=c99 
MKL_LINKING_FLAGS=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
MKL_OPENMP_LINKING_FLAGS= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

CC=mpiicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

test_mpi_io: test_mpi_io.c
	${VERBOSE} ${CC} -o test_mpi_io test_mpi_io.c ${CFLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}


debug: lu.c lu.h
	${VERBOSE} ${CC} -g -o lu lu.c ${DEBUG_FLAGS} $(MKL_LINKING_FLAGS) -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 
smol: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -D SMALL_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

mini: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -D MINI_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 


medium: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -D MEDIUM_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

large: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -D LARGE_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 


extralarge: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} $(MKL_LINKING_FLAGS) -D EXTRALARGE_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

clean:
	@ rm -f lu

