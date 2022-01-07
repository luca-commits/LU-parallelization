/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Array initialization. */
static void init_array(int n, double **A, double **B) {
  int i, j;

#pragma omp parallel for collapse(2) private(i, j)
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data
   Can be used also to check the correctness of the output. */
static void print_array(int n, double **A)

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int tsteps, int n, double **A, double **B) {
  int t, i, j;

  for (t = 0; t < _PB_TSTEPS; t++) {
#pragma omp parallel for collapse(2)
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] +
                                     A[1 + i][j] + A[i - 1][j]);
#pragma omp parallel for collapse(2)
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] +
                                     B[1 + i][j] + B[i - 1][j]);
  }
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int runs, n, tsteps;

  if (argc == 4) {
    runs = atoi(argv[1]);
    n = atoi(argv[2]);
    tsteps = atoi(argv[3]);

#define N n
#define _PB_TSTEPS tsteps
  } else {
    printf(
        "Wrong number of arguments provided!\nUSAGE: Arg 1: No of runs\n     "
        "  Arg 2: Problem size N\n"
        "  Arg 3: Time steps\n");
    return -1;
  }
  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  // POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  double **A = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) A[i] = (double *)malloc(n * sizeof(double));

  double **B = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) B[i] = (double *)malloc(n * sizeof(double));

  double timings[runs];
  double start_time;

  for (int i = 0; i < runs; ++i) {
    /* Initialize array(s). */
    init_array(n, A, B);

    /* Start timer. */
    start_time = omp_get_wtime();

    /* Run kernel. */
    kernel_jacobi_2d(tsteps, n, A, B);

    /* Stop timer. */
    timings[i] = omp_get_wtime() - start_time;
  }

  int size = omp_get_max_threads();

  char *filename;
  asprintf(&filename, "%d.csv", size);
  FILE *timings_file = fopen(filename, "w");

  fprintf(timings_file, "n,tsteps,p,time,value\n");

  for (int i = 0; i < runs; ++i) {
    fprintf(timings_file, "%d,%d,%d,%f,%f\n", n, tsteps, size, timings[i], 0.0);
  }

  fclose(timings_file);

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
