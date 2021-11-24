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

inline unsigned phi0(unsigned k, unsigned distr_M) { return k % distr_M; }

inline unsigned phi1(unsigned k, unsigned distr_N) { return k % distr_N; }

inline unsigned P(unsigned s, unsigned t, unsigned distr_M, unsigned distr_N) {
  return s % distr_M + t * distr_M;
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
  unsigned nr = (n + distr_M - s - 1) / distr_M;
  int i, j, k, r;
  DATA_TYPE absmax;
  int dummy;
  DATA_TYPE pivot;

  unsigned pi_k_temp, pi_r_temp;
  unsigned A_row_k_temp[n / distr_N], A_row_r_temp[n / distr_N];
  unsigned counter = 0;  // used to see how many entries A_row_k_temp will have

  // find largest absolute value in column k
  DATA_TYPE* Max = (DATA_TYPE*)malloc(MAX(distr_M, 1) * sizeof(DATA_TYPE));
  int* IMax = (int*)malloc(MAX(distr_M, 1) * sizeof(int));

  for (k = 0; k < n; k++) {
    int kr = (k + distr_M - s - 1) / distr_M;
    int kc = (k + distr_N - t - 1) / distr_N;

    if (phi1(k, distr_N) == t) {
      

      absmax = A[0][k];
      int imax = 0;

      //printf("%d\n", kr);
      for (i = kr; i < nr; i++) {
        if (absmax < fabs(A[i][kc])) {
          absmax = fabs(A[i][kc]);

          imax = i;
        }
      }


      double max = 0;
      if (absmax > 0) {
        printf("%d %d\n", imax, kc);
        max = A[imax][kc];
      }
      printf("hello\n");

      // TODO change second argument -> count

      // MPI_Request requests[4 * distr_M];

      // for (i = 0; i < distr_M; ++i) {
      //   MPI_Isend(&max, 1, MPI_DOUBLE, P(i, t, distr_M, distr_N), 0,
      //             MPI_COMM_WORLD, &requests[2 * i]);
      //   MPI_Isend(&imax, 1, MPI_INT, P(i, t, distr_M, distr_N), 0,
      //             MPI_COMM_WORLD, &requests[2 * i + 1]);
      // }

      // for (i = 0; i < distr_M; ++i) {
      //   // MPI_Irecv(&Max[i], 1, MPI_DOUBLE, P(i, t, distr_M, distr_N), 0,
      //   //           MPI_COMM_WORLD, &requests[2 * distr_M + 2 * i]);
      //   // MPI_Irecv(&IMax[i], 1, MPI_DOUBLE, P(i, t, distr_M, distr_N), 0,
      //   //           MPI_COMM_WORLD, &requests[2 * distr_M + 2 * i + 1]);

      //   printf("%d\n", P(i, t, distr_M, distr_N));
      // }

      // MPI_Waitall(4 * distr_M, requests, MPI_STATUSES_IGNORE);
    }

    printf("finished superstep 1\n");

    // // superstep 2 & 3
    // if (phi1(k, distr_N) == t) {
    //   absmax = 0;
    //   dummy = 0;
    //   // n mit M ersetzen?
    //   for (i = 0; i < distr_M; i++) {
    //     if (fabs(Max[i]) > absmax) {
    //       absmax = fabs(Max[i]);
    //       dummy = i;
    //     }
    //   }
    //   // define EPS 1.0e-15?
    //   if (absmax > 1.03e-15) {
    //     int imax = IMax[dummy];
    //     // n mit M ersetzen?
    //     r = imax * distr_M + dummy;  // global index
    //     pivot = Max[dummy];
    //     for (j = kr; j < nr; j++) {
    //       A[j][kc] /= pivot;
    //     }
    //     if (s == dummy) {
    //       A[dummy][kc] = pivot;
    //     }
    //   } else {
    //     // MPI_Abort(MPI_COMM_WORLD, int singularmatrix);
    //   }

    //   /* Superstep (3) */
    //   MPI_Request requests[distr_N - 1];

    //   for (i = 0; i < distr_N; ++i) {
    //     if (t != i)
    //       MPI_Isend(&r, 1, MPI_DOUBLE, P(s, i, distr_M, distr_N), 0,
    //                 MPI_COMM_WORLD, &requests[i]);
    //   }

    //   MPI_Waitall(distr_N - 1, requests, MPI_STATUSES_IGNORE);
    // }

    // if (s == phi0(k, distr_N)) {
    //   MPI_Recv(&r, 1, MPI_DOUBLE, P(s, phi1(k, distr_N), distr_M, distr_N),
    //   0,
    //            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    // printf("%d\n", r);

    // /* Superstep (4) & ... */
    // if (k % distr_M == s && r != k) {
    //   /* Store pi(k) in pi(r) on P(r%M,t) */
    //   MPI_Send(&pi[k / distr_M], 1, MPI_DOUBLE, r % distr_M + t * distr_M, k,
    //            MPI_COMM_WORLD);
    //   for (j = 0; j < n; ++j) {  // waistful looping... will correct later
    //     if (j % distr_N == t) {
    //       A_row_k_temp[counter] =
    //           A[k][j]; /*counter was set to 0 at the beginning of the
    //                      function, used to see what size the buffer will have
    //                      (and doubles index here)
    //                     */
    //       counter++;
    //     }
    //   }
    //   /*I'm not sure if reciever would get right write message without tag so
    //    * I just added one */
    //   MPI_Send(&A_row_k_temp, counter, MPI_DOUBLE, r % distr_M + t * distr_M,
    //            k + 1, MPI_COMM_WORLD);  // Probably should use ISend here (or
    //                                     // even use one sided communication)
    // }

    // if (p_id == r % distr_M + t * distr_M) {
    //   MPI_Recv(&pi_k_temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
    //            MPI_STATUS_IGNORE);
    //   MPI_Recv(&A_row_k_temp, counter, MPI_DOUBLE, MPI_ANY_SOURCE, k + 1,
    //            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    // if (r % distr_M == s && r != k) {
    //   i = 0;
    //   MPI_Send(&pi[r / distr_M], 1, MPI_DOUBLE, k % distr_M + t * distr_M,
    //            k + 2, MPI_COMM_WORLD);
    //   for (j = 0; j < n; ++j) {
    //     if (j % distr_N == t) {
    //       A_row_r_temp[i] = A[r][j];  // should be able to reuse counter (but
    //                                   // need extra index variable i)
    //     }
    //   }
    //   MPI_Send(&A_row_r_temp, counter, MPI_DOUBLE, r % distr_M + t * distr_M,
    //            k + 3, MPI_COMM_WORLD);
    // }

    // if (p_id == k % distr_M + t * distr_M) {
    //   MPI_Recv(&pi_r_temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k + 2,
    //   MPI_COMM_WORLD,
    //            MPI_STATUS_IGNORE);
    //   MPI_Recv(&A_row_r_temp, counter, MPI_DOUBLE, MPI_ANY_SOURCE, k + 3,
    //            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    // if (k % distr_M == s) {
    //   pi[k] = pi_r_temp;
    //   i = 0;
    //   for (j = t; j < n; j += distr_N) {
    //     A[k][j] = A_row_r_temp[i];
    //     ++i;
    //   }
    // }

    // if (r % distr_M == s) {
    //   pi[r] = pi_k_temp;
    //   i = 0;
    //   for (j = t; j < n; j += distr_N) {
    //     A[r][j] = A_row_k_temp[i];
    //     ++i;
    //   }
    // }

    // // algo 2.4 begin

    // // superstep 8
    // int size;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if (k % distr_M == s && k % distr_N == t) {
    //   for (i = 0; i < distr_M; ++i) {
    //     MPI_Send(&A[k][k], 1, MPI_DOUBLE, P(i, t, distr_M, distr_N), k,
    //              MPI_COMM_WORLD);
    //   }
    // }

    // unsigned a_kk;

    // if (phi1(k, distr_N) == t) {
    //   for (i = 0; i < distr_N; ++i) {
    //     if (p_id == P(i, t, distr_M, distr_N)) {
    //       MPI_Recv(&a_kk, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
    //                MPI_STATUS_IGNORE);
    //     }
    //   }
    // }

    // // superstep 9

    // if (phi1(k, distr_N) == t) {
    //   for (i = k; i < n; ++i) {
    //     if (phi0(i, distr_M) == s) {
    //       A[i][k] /= a_kk;
    //     }
    //   }
    // }

    // // superstep 10
    // if (phi1(k, distr_N) == t) {
    //   for (i = k; i < n; ++i) {
    //     if (phi0(i, distr_N) == s) {
    //       for (j = 0; j < distr_N; ++j) {
    //         MPI_Send(&A[i][k], 1, MPI_DOUBLE, P(s, j, distr_M, distr_N), k,
    //                  MPI_COMM_WORLD);
    //       }
    //     }
    //   }
    // }

    // unsigned a_ik;
    // if (phi1(k, distr_N) != t) {
    //   MPI_Recv(&a_ik, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
    //            MPI_STATUS_IGNORE);
    // }

    // if (phi0(k, distr_M) == s) {
    //   for (j = k; j < n; ++j) {
    //     if (phi1(j, distr_N) == t) {
    //       for (i = 0; i < distr_M; ++i) {
    //         MPI_Send(&A[k][j], 1, MPI_DOUBLE, P(j, t, distr_M, distr_N), k,
    //                  MPI_COMM_WORLD);
    //       }
    //     }
    //   }
    // }

    // unsigned a_kj;
    // if (phi0(k, distr_M) != s) {
    //   MPI_Recv(&a_kj, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
    //            MPI_STATUS_IGNORE);
    // }

    // // superstep 11
    // for (i = k; i < n; ++i) {
    //   if (phi0(i, distr_M) == s) {
    //     for (j = 0; j < n; ++j) {
    //       if (phi1(j, distr_N) == t) {
    //         A[i][j] -= a_ik * a_kj;
    //       }
    //     }
    //   }
    // }
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
  unsigned i;
  for (i = 0; i < n; ++i) {
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
  if (rank == 0)
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  MPI_Finalize();

  return 0;
}
