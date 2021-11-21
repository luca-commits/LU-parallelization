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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Include MPI header. */
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
    for (j = i + 1; j < n; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r, s, t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s) (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s) (POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s) A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

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

/* Swap functions */
static void swap(DATA_TYPE x, DATA_TYPE y) {
  DATA_TYPE temp = x;
  x = y;
  y = temp;
}

/* Helper functions:
   Index mapping to processors
*/
static unsigned phi0(unsigned i, unsigned distr_M) { return i % distr_M; }
static unsigned phi1(unsigned j, unsigned distr_N) { return j % distr_N; }
static unsigned get_rank(unsigned s, unsigned t, unsigned distr_M) {
  return s + distr_M * t;
};

/* Main computational kernel. The whole function will be timed,
   including the call and return.
   pi: permutation vector (need to solve lse)
   distr_M, distr_N: rows and cols of the cyclic distribution
*/
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                      unsigned p_id, unsigned* pi, unsigned distr_M,
                      unsigned distr_N) {
  unsigned s = p_id % distr_M;
  unsigned t = p_id / distr_M;
  DATA_TYPE nr = (n + _PB_N - s - 1) / _PB_N;
  DATA_TYPE nc = (n + _PB_N - t - 1) / _PB_N;
  int i, j, k, r;
  int p[_PB_N];
  DATA_TYPE absmax;
  DATA_TYPE max;

  unsigned pi_k_temp, pi_r_temp;
  unsigned A_row_k_temp[n / distr_N], A_row_r_temp[n / distr_N];
  unsigned counter = 0;  // used to see how many entries A_row_k_temp will have

#pragma scop

  // find largest absolute value in column k
  DATA_TYPE* Max = malloc(MAX(_PB_N, 1) * sizeof(DATA_TYPE));
  DATA_TYPE* IMax = malloc(MAX(_PB_N, 1) * sizeof(long));
  for (k = 0; k < _PB_N; k++) {
    /* Superstep (4), (6) send for phi_0(k) == s */
    if (phi0(k, distr_M) == s && r != k) {
      /* Store pi(k) in pi(r) on P(r%M,t) */
      /* Send permutation vector element */
      MPI_Send(&pi[k / distr_M], 1, MPI_DOUBLE,
               get_rank(phi0(r, distr_M), t, distr_M), k, MPI_COMM_WORLD);
      for (unsigned j = 0; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          A_row_k_temp[counter] =
              A[k][j]; /*counter was set to 0 at the beginning of the function,
                         used to see what size the buffer will have (and doubles
                         index here)
                        */
          counter++;
        }
      }

      /* Send elements of A */
      MPI_Send(&A_row_k_temp, counter, MPI_DOUBLE,
               get_rank(phi0(r, distr_M), t, distr_M), k + 1,
               MPI_COMM_WORLD);  // Probably should use ISend here (or
                                 // even use one sided communication)
    }

    /* Superstep (4), (6) receive for phi_0(r) == s */
    if (get_rank(phi0(r, distr_M), t, distr_M) == p_id) {
      MPI_Recv(&pi_k_temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&A_row_k_temp, counter, MPI_DOUBLE, MPI_ANY_SOURCE, k + 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Superstep (4), (6) send for phi_0(r) == s */
    if (phi0(r, distr_M) == s && r != k) {
      unsigned i = 0;
      MPI_Send(&pi[r / distr_M], 1, MPI_DOUBLE,
               get_rank(phi0(k, distr_M), t, distr_M), k + 2, MPI_COMM_WORLD);
      for (unsigned j = 0; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          A_row_r_temp[i] = A[r][j];  // should be able to reuse counter (but
                                      // need extra index variable i)
          ++i;
        }
      }
      MPI_Send(&A_row_r_temp, counter, MPI_DOUBLE,
               get_rank(phi0(r, distr_M), t, distr_M), k + 3, MPI_COMM_WORLD);
    }

    /* Superstep (4), (6) receive for phi_0(k) == */
    if (p_id == get_rank(phi0(k, distr_M), t, distr_M)) {
      MPI_Recv(&pi_r_temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k + 2, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&A_row_r_temp, counter, MPI_DOUBLE, MPI_ANY_SOURCE, k + 3,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Superstep (7) */
    if (phi0(k, distr_M) == s) {
      pi[k] = pi_r_temp;
      unsigned idx_counter = 0;
      for (unsigned j = 0; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          A[k][j] = A_row_r_temp[idx_counter];
          ++idx_counter;
        }
      }
    }

    if (phi0(r, distr_M) == s) {
      pi[r] = pi_k_temp;
      unsigned idx_counter = 0;
      for (unsigned j = 0; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          A[r][j] = A_row_k_temp[idx_counter];
          ++idx_counter;
        }
      }
    }

    // memory-efficient sequential LU decomposition
    for (i = k + 1; i < _PB_N; i++) {
      A[i][k] /= A[k][k];
    }
    for (i = k + 1; i < _PB_N; i++) {
      for (j = k + 1; j < _PB_N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
#pragma endscop
  }
}

int main(int argc, char** argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  unsigned* pi = malloc(sizeof(unsigned) * n);
  for (unsigned i = 0; i < n; ++i) {
    pi[i] = i;
  }

  unsigned distr_M,
      distr_N;  // M and N of the cyclic distr., need to be computed yet

  distr_M = largest_divisor(size);
  distr_N = size / distr_M;

  /* Run kernel. */
  kernel_lu(n, POLYBENCH_ARRAY(A), rank, pi, distr_M, distr_N);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}
