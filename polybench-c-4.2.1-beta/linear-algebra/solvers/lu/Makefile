include ../../../config.mk

EXTRA_FLAGS=


CC=mpicc

lu: lu.c lu.h
	${VERBOSE} ${CC} -o lu lu.c ${CFLAGS} -I. -I../../../utilities ../../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f lu

