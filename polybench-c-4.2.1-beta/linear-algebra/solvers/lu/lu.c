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

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), p_id) {
  DATA_TYPE s = p_id % _PB_N;
  DATA_TYPE t = p_id / _PB_N;
  DATA_TYPE nr = (n + _PB_N - s - 1) / _PB_N;
  DATA_TYPE nc = (n + _PB_N - t - 1) / _PB_N;
  int i, j, k, r;
  int p[_PB_N];
  DATA_TYPE absmax;
  DATA_TYPE max;

#pragma scop


  // find largest absolute value in column k
  DATA_TYPE* Max = malloc(MAX(_PB_N, 1) * sizeof(DATA_TYPE));
  DATA_TYPE* IMax = malloc(MAX(_PB_N, 1) * sizeof(long));
  for (k = 0; k < _PB_N; k++) {
    DATA_TYPE kr = (k + _PB_N - s - 1) / _PB_N;
    DATA_TYPE kc = (k + _PB_N - t - 1) / _PB_N;
    if (k % _PB_N == t) {
      absmax = A[0][k];
      for (i = kr; i < nr; i++) {
        if (absmax < fabs(A[i][kc])) {
          absmax = fabs(A[i][kc]);
          r = i;
        }
      }
      max = 0;
      if (absmax > 0.0) {
        max = A[r][kc]
      }
      for (j = 0; j < _PB_N; j++) {
        // Probably wrong ...
        MPI_Bcast(j + t * _PB_N, &max, Max, s * sizeof(DATA_TYPE),
                  sizeof(DATA_TYPE));
        MPI_Bcast(j + t * _PB_N, &r, IMax, s * sizeof(long), sizeof(long));
      }
    }
    R[k-k0]= r; /* store index of pivot row */

      long nperm= 0;
      long Src2[2], Dest2[2];
      if (k%M==s && r!=k){
          /* Store pi(k) in pi(r) on P(r%M,t) */
          MPI_Send(&)
          Src2[nperm]= k; Dest2[nperm]= r; nperm++;
      }
      if (r%M==s && r!=k){
          bsp_put(k%M+t*M,&pi[r/M],pi,(k/M)*sizeof(long),
                  sizeof(long));
          Src2[nperm]= r; Dest2[nperm]= k; nperm++;
      }
      /* Swap rows k and r for columns in range k0..k0+b-1 */
      bsp_permute_rows(M,Src2,Dest2,nperm,pa,nc,k0c,k0cb);
      bsp_sync();

      /****** Superstep (6) ******/
      /* Phase 0 of two-phase broadcasts */
      if (k%N==t){ 
          /* Store new column k in Lk */
          for (long i=kr1; i<nr; i++)     
              Lk[i-kr1]= a[i][kc];
      }
      if (k%M==s){ 
          /* Store new row k in Uk for columns
              in range k+1..k0+b-1 */
          for (long j=kc1; j<k0cb; j++)
              Uk[j-kc1]= a[kr][j];
      }
      bsp_broadcast(Lk,nr-kr1,s+(k%N)*M,s,M,N,0);
      bsp_broadcast(Uk,k0cb-kc1,(k%M)+t*M,t*M,1,M,0);
      bsp_sync();
      
      /****** Superstep (7) ******/
      /* Phase 1 of two-phase broadcasts */
      bsp_broadcast(Lk,nr-kr1,s+(k%N)*M,s,M,N,1); 
      bsp_broadcast(Uk,k0cb-kc1,(k%M)+t*M,t*M,1,M,1);
      bsp_sync();
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

  /* Run kernel. */
  kernel_lu(n, POLYBENCH_ARRAY(A), &rank);

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
