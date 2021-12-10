include ../../../config.mk

EXTRA_FLAGS=

lu: lu.c lu.h
	${VERBOSE} ${CC} -fopenmp -o lu lu.c ${CFLAGS} -D SMALL_DATASET -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}


clean:
	@ rm -f lu

