/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Include MPI header. */
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define EPS 1.0e-15

extern void blacs_get_(int *, int *, int *);
extern void blacs_pinfo_(int *, int *);
extern void blacs_gridinit_(int *, char *, int *, int *);
extern void blacs_gridinfo_(int *, int *, int *, int *, int *);
extern void descinit_(int *, int *, int *, int *, int *, int *, int *, int *,
                      int *, int *);
extern void pdgetrf_(int *, int *, double *, int *, int *, int *, int *, int *);
extern void pdpotrf_(char *, int *, double *, int *, int *, int *, int *);
extern void blacs_gridexit_(int *);
extern int numroc_(int *, int *, int *, int *, int *);

unsigned phi0(unsigned i, unsigned distr_M) { return i % distr_M; }

unsigned phi1(unsigned j, unsigned distr_N) { return j % distr_N; }

unsigned idx(unsigned i, unsigned j, unsigned nr) { return j * nr + i; }

unsigned i_loc(unsigned i, unsigned distr_M) { return i / distr_M; }

unsigned j_loc(unsigned j, unsigned distr_N) { return j / distr_N; }

unsigned i_glob(unsigned i, unsigned distr_M, unsigned s) {
  return i * distr_M + s;
}

unsigned j_glob(unsigned j, unsigned distr_N, unsigned t) {
  return j * distr_N + t;
}

// could be a macro for efficiency
// source:
// https://slaystudy.com/c-c-program-to-find-the-largest-divisor-of-a-number/
int largest_divisor(int n) {
  int i;
  for (i = n - 1; i >= 1; --i) {
    // if i divides n, then i is the largest divisor of n
    // return i
    if (n % i == 0) return i;
  }
  // we reach this point if n is equal to 1
  // there is no proper divisor of 1
  // simply return 0
  return 0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nr, int nc, double *A, unsigned distr_M,
                        unsigned distr_N) {
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  fprintf(POLYBENCH_DUMP_TARGET, "\n");
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[idx(i, j, nr)]);
    }

    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Array initialization. */
// check this one!!1!!1!!
static void init_array(int n, int nr, int nc, unsigned distr_M,
                       unsigned distr_N, double *A, unsigned s, unsigned t,
                       unsigned p_id /* for debugging*/) {
  // printf("rank=%d s=%d t=%d\n", p_id, s, t);

  for (unsigned j = 0; j < nc; ++j) {
    for (unsigned i = 0; i < nr; ++i) {
      if (j_glob(j, distr_N, t) < i_glob(i, distr_M, s)) {
        A[idx(i, j, nr)] = ((double)(-j_glob(j, distr_N, t) % n) / n + 1) * n;
      } else if (i_glob(i, distr_M, s) == j_glob(j, distr_N, t)) {
        A[idx(i, j, nr)] = 1 * n;
      } else {
        A[idx(i, j, nr)] = 0;
      }
      // A[idx(i, j, nc)] = (double)(rand()) / RAND_MAX * 2.;
    }
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return.
   pi: permutation vector (need to solve lse)
   distr_M, distr_N: rows and cols of the cyclic distribution
*/
static void kernel_lu(int n, double *A, unsigned p_id, unsigned s, unsigned t,
                      unsigned *pi, const unsigned distr_M,
                      const unsigned distr_N, MPI_Comm comm_row,
                      MPI_Comm comm_col) {}

int main(int argc, char **argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n, runs;
  if (argc == 3) {
    runs = atoi(argv[1]);
    n = atoi(argv[2]);
  } else {
    if (rank == 0)
      printf(
          "Wrong number of arguments provided!\nUSAGE: Arg 1: No of runs\n     "
          "  Arg 2: Problem size N\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int distr_M,
      distr_N;  // M and N of the cyclic distr., need to be computed yet

  if (size != 1) {
    distr_M = largest_divisor(size);
    distr_N = size / distr_M;
  } else {
    distr_M = 1;
    distr_N = 1;
  }

  int s, t;
  int iam, nprocs;
  int zero = 0;
  int one = 1;
  int ictxt;
  char layout = 'R';  // Block cyclic, Row major processor mapping

  blacs_pinfo_(&iam, &nprocs);
  blacs_get_(&zero, &zero, &ictxt);
  blacs_gridinit_(&ictxt, &layout, &distr_M, &distr_N);
  blacs_gridinfo_(&ictxt, &distr_M, &distr_N, &s, &t);

  printf("rank %d: s=%d t=%d\n", rank, s, t);

  int nr = numroc_(&n, &one, &s, &zero, &distr_M);
  int nc = numroc_(&n, &one, &t, &zero, &distr_N);

  printf(
      "Hi. Proc %d/%d for MPI, proc %d/%d for BLACS in position "
      "(%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size"
      "%d\n",
      rank, size, iam, nprocs, s, t, distr_M, distr_N, nr, nc, n, 1);

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  double *A = (double *)malloc(nr * nc * sizeof(double));
  int *pi = (int *)malloc(sizeof(int) * n);

  double timings[runs];

  /* Initialize array(s). */
  for (int i = 0; i < runs; ++i) {
    init_array(n, nr, nc, distr_M, distr_N, A, s, t, rank);
    // if (rank == 0) print_array(nr, nc, A, distr_M, distr_N);

    int desc[9];
    int info;
    descinit_(desc, &n, &n, &one, &one, &zero, &zero, &ictxt, &nr, &info);
    if (info != 0) {
      printf("Error in descinit, info = %d\n", info);
    }

    /* Start timer. */
    double start_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    /* Run kernel. */
    MPI_Pcontrol(1, "Kernel");
    pdgetrf_(&n, &n, A, &one, &one, desc, pi, &info);
    if (info != 0) {
      printf("Error in pdgemm, info = %d\n", info);
    }
    MPI_Pcontrol(-1, "Kernel");

    /* Stop and print timer. */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) timings[i] = MPI_Wtime() - start_time;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    // if (rank == 0) polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
  }

  if (rank == 0) {
    char *filename;
    asprintf(&filename, "%d.csv", size);
    FILE *timings_file = fopen(filename, "w");

    fprintf(timings_file, "n,p,time,value\n");

    for (int i = 0; i < runs; ++i) {
      fprintf(timings_file, "%d,%d,%f,%f\n", n, size, timings[i], 0.0);
    }

    fclose(timings_file);
  }

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  blacs_gridexit_(&ictxt);
  MPI_Finalize();

  return 0;
}
