include ../../../config.mk

EXTRA_FLAGS=

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

