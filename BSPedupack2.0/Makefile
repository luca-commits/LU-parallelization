CC= bspcc
CFLAGS= -std=c99 -Wall -O3
LFLAGS= -lm

OBJIP= bspinprod.o bspedupack.o
OBJBEN= bspbench.o bspedupack.o
OBJSORT= bspsort_test.o bspsort.o bspedupack.o
OBJLU= bsplu_test.o bsplu.o bspedupack.o
OBJFFT= bspfft_test.o bspfft.o bspedupack.o
OBJMV= bspmv_test.o bspmv.o bspsparse_input.o bspedupack.o
OBJMATCH= bspmatch_test.o bspmatch.o bspsparse_input.o bspedupack.o

all: inprod bench sort lu fft matvec match

inprod: $(OBJIP)
	$(CC) $(CFLAGS) -o inprod $(OBJIP) $(LFLAGS)

bench: $(OBJBEN)
	$(CC) $(CFLAGS) -o bench $(OBJBEN) $(LFLAGS)

sort: $(OBJSORT)
	$(CC) $(CFLAGS) -o sort $(OBJSORT) $(LFLAGS)

lu: $(OBJLU)
	$(CC) $(CFLAGS) -o lu $(OBJLU) $(LFLAGS)

fft: $(OBJFFT)
	$(CC) $(CFLAGS) -o fft $(OBJFFT) $(LFLAGS)

matvec: $(OBJMV)
	$(CC) $(CFLAGS) -o matvec $(OBJMV) $(LFLAGS)

match: $(OBJMATCH)
	$(CC) $(CFLAGS) -o match $(OBJMATCH) $(LFLAGS)

bspinprod.o:  bspedupack.h
bspbench.o:   bspedupack.h
bspsort.o:    bspedupack.h
bspsort_test.o:    bspedupack.h
bsplu.o:    bspedupack.h
bsplu_test.o:    bspedupack.h
bspfft.o:    bspedupack.h
bspfft_test.o:    bspedupack.h
bspmv.o:    bspedupack.h
bspmv_test.o:    bspedupack.h bspsparse_input.h
bspmatch_test.o:    bspedupack.h bspsparse_input.h
bspsparse_input.o: bspedupack.h bspsparse_input.h
bspedupack.o: bspedupack.h

.PHONY: clean
clean:
	rm -f $(OBJIP) $(OBJBEN) $(OBJSORT) $(OBJLU) $(OBJFFT) $(OBJMV) $(OBJMATCH) inprod  bench sort lu fft matvec match
