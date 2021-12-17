include ../../../config.mk

EXTRA_FLAGS=-Wall -DMINI_DATASET -std=c99 -DPOLYBENCH_TIME


CC=mpiicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

test_mpi_io: test_mpi_io.c
	${VERBOSE} ${CC} -o test_mpi_io test_mpi_io.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}


debug: lu.c lu.h
	${VERBOSE} ${CC} -g -o lu lu.c ${DEBUG_FLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 
smol: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -D SMALL_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

mini: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -D MINI_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 


medium: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -D MEDIUM_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

large: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -D LARGE_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 


extralarge: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -D EXTRALARGE_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

clean:
	@ rm -f lu

