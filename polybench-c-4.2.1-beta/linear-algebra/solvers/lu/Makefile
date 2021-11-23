include ../../../config.mk

EXTRA_FLAGS=-Wall -DMINI_DATASET


CC=mpicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}


debug: lu.c lu.h
	${VERBOSE} ${CC} -g -o lu lu.c -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS} 

clean:
	@ rm -f lu

