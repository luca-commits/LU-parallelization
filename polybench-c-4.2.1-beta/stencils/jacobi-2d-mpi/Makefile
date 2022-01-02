include ../../config.mk

CC=mpicc
EXTRA_FLAGS=-Wall -DPOLYBENCH_TIME -std=c99 -lm -D SMALL_DATASET

jacobi-2d: jacobi-2d.c jacobi-2d.h
	${VERBOSE} ${CC} -o jacobi-2d jacobi-2d.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

jacobi-2d-original: jacobi-2d-original.c jacobi-2d.h
	${VERBOSE} ${CC} -o jacobi-2d-original jacobi-2d-original.c ${CFLAGS} -I. -I../../utilities ../../utilities/polybench.c ${EXTRA_FLAGS}

clean:
	@ rm -f jacobi-2d jacobi-2d-original

