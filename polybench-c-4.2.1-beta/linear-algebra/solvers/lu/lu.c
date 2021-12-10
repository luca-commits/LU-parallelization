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

#define EPS 1.0e-15

unsigned phi0(unsigned i, unsigned distr_M) { return i % distr_M; }

unsigned phi1(unsigned j, unsigned distr_N) { return j % distr_N; }

unsigned idx(unsigned i, unsigned j, unsigned nc) { return i * nc + j; }

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
static void print_array(int nr, int nc, double* A, unsigned distr_M,
                        unsigned distr_N) {
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  fprintf(POLYBENCH_DUMP_TARGET, "\n");
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[idx(i, j, nc)]);
    }

    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Array initialization. */
// check this one!!1!!1!!
static void init_array(int n, int nr, int nc, unsigned distr_M,
                       unsigned distr_N, double* A, unsigned s, unsigned t,
                       unsigned p_id /* for debugging*/) {
  printf("rank=%d s=%d t=%d\n", p_id, s, t);

  for (unsigned i = 0; i < nc; ++i) {
    for (unsigned j = 0; j < nr; ++j) {
      if (j_glob(j, distr_N, t) < i_glob(i, distr_M, s)) {
        A[idx(i, j, nc)] = (double)(-j_glob(j, distr_N, t) % n) / n + 1;
      } else if (i_glob(i, distr_M, s) == j_glob(j, distr_N, t)) {
        A[idx(i, j, nc)] = 1;
      } else {
        A[idx(i, j, nc)] = 0;
      }
    }
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return.
   pi: permutation vector (need to solve lse)
   distr_M, distr_N: rows and cols of the cyclic distribution
*/
static void kernel_lu(int n, double* A, unsigned p_id, unsigned* pi,
                      unsigned distr_M, unsigned distr_N) {
  unsigned t = p_id % distr_M;  // have to check these ones out!!!!!!
  unsigned s = p_id / distr_M;
  // unsigned nr = (n + distr_M - s - 1) / distr_M;  // number of local rows
  unsigned nr = (n + distr_M - s - 1) / distr_M;
  unsigned nc = (n + distr_N - t - 1) / distr_N;  // number of local columns
  int i, j, k, r;
  DATA_TYPE absmax;

  unsigned pi_k_temp, pi_r_temp;
  double A_row_k_temp[nc];
  double A_row_r_temp[nc];

  printf("pid: %d \n", p_id);
  // algorithm 2.5

  // find largest absolute value in column k
  DATA_TYPE* Max = (DATA_TYPE*)malloc(MAX(distr_M, 1) * sizeof(DATA_TYPE));
  int* IMax = (int*)malloc(MAX(distr_M, 1) * sizeof(int));

  for (k = 0; k < n; k++) {
    if (phi1(k, distr_N) == t) {
      absmax = A[idx(i_loc(0, distr_M), j_loc(k, distr_N), nc)];
      int rs = 0;

      for (i = k; i < n; i++) {
        if (phi0(i, distr_N) == s &&
            absmax < fabs(A[idx(i_loc(i, distr_M), j_loc(k, distr_N), nc)])) {
          absmax = fabs(A[idx(i_loc(i, distr_M), j_loc(k, distr_N), nc)]);
          rs = i;
        }
      }

      double max = 0;
      if (absmax > EPS) {
        max = A[idx(i_loc(rs, distr_M), j_loc(k, distr_N), nc)];
      }

      MPI_Request requests[4 * distr_M];

      for (i = 0; i < distr_M; ++i) {
        MPI_Isend(&max, 1, MPI_DOUBLE, i * distr_M + t, 0, MPI_COMM_WORLD,
                  &requests[2 * i]);
        MPI_Isend(&rs, 1, MPI_INT, i * distr_M + t, 0, MPI_COMM_WORLD,
                  &requests[2 * i + 1]);
      }

      for (i = 0; i < distr_M; ++i) {
        MPI_Irecv(&Max[i], 1, MPI_DOUBLE, i * distr_M + t, 0, MPI_COMM_WORLD,
                  &requests[2 * distr_M + 2 * i]);
        MPI_Irecv(&IMax[i], 1, MPI_DOUBLE, i * distr_M + t, 0, MPI_COMM_WORLD,
                  &requests[2 * distr_M + 2 * i + 1]);
      }

      MPI_Waitall(4 * distr_M, requests, MPI_STATUSES_IGNORE);
    }

    // superstep 2 & 3
    if (phi1(k, distr_N) == t) {
      absmax = 0;
      unsigned smax = 0;

      for (i = 0; i < distr_M; i++) {
        if (fabs(Max[i]) > absmax) {
          absmax = fabs(Max[i]);
          smax = i;
        }
      }

      if (absmax > EPS) {
        int imax = IMax[smax];
        r = imax;  // global index
      } else {
        printf("rank %d: ABORT because of pivoting on zero element\n", p_id);
        MPI_Abort(MPI_COMM_WORLD, 192);
      }
      /* Superstep (3) */
      MPI_Request requests[distr_N];

      for (i = 0; i < distr_N; ++i) {
        if (t != i) {
          MPI_Isend(&r, 1, MPI_DOUBLE, distr_M * s + i, 0, MPI_COMM_WORLD,
                    &requests[i]);
        } else
          requests[i] = MPI_REQUEST_NULL;
      }

      MPI_Waitall(distr_N, requests, MPI_STATUSES_IGNORE);
    }

    if (t != phi1(k, distr_N)) {
      MPI_Recv(&r, 1, MPI_DOUBLE, s * distr_M + phi1(k, distr_N), 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("rank %d: swap row k=%d with row r=%d\n", p_id, k, r);

    /* Superstep (4) & ... */
    if (phi0(k, distr_M) == s && r != k) {
      /* Store pi(k) in pi(r) on P(r%M,t) */
      MPI_Send(&pi[k / distr_M], 1, MPI_DOUBLE, distr_M * phi0(r, distr_M) + t,
               k, MPI_COMM_WORLD);

      i = 0;
      for (j = 0; j < n; ++j) {  // waistful looping... will correct later
        if (phi1(j, distr_N) == t) {
          A_row_k_temp[i] = A[idx(i_loc(k, distr_M), j_loc(j, distr_N), nc)];
          ++i;
        }
      }
      /*I'm not sure if reciever would get right write message without tag so
       * I just added one */
      MPI_Send(&A_row_k_temp, nc, MPI_DOUBLE, distr_M * phi0(r, distr_M) + t,
               k + 1,
               MPI_COMM_WORLD);  // Probably should use ISend here (or
                                 // even use one sided communication)
    }

    if (s == phi0(r, distr_M) && r != k) {
      MPI_Recv(&pi_k_temp, 1, MPI_DOUBLE, phi0(k, distr_M) * distr_M + t, k,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Recv(&A_row_k_temp, nc, MPI_DOUBLE, phi0(k, distr_M) * distr_M + t,
               k + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (phi0(r, distr_M) == s && r != k) {
      i = 0;
      MPI_Send(&pi[r / distr_M], 1, MPI_DOUBLE, distr_M * phi0(k, distr_M) + t,
               k + 2, MPI_COMM_WORLD);
      for (j = 0; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          A_row_r_temp[i] = A[idx(i_loc(r, distr_M), j_loc(j, distr_N), nc)];
          i++;
        }
      }
      MPI_Send(&A_row_r_temp, nc, MPI_DOUBLE, phi0(k, distr_M) * distr_M + t,
               k + 3, MPI_COMM_WORLD);
    }

    if (s == phi0(k, distr_M) && r != k) {
      MPI_Recv(&pi_r_temp, 1, MPI_DOUBLE, phi0(r, distr_M) * distr_M + t, k + 2,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&A_row_r_temp, nc, MPI_DOUBLE, phi0(r, distr_M) * distr_M + t,
               k + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (phi0(k, distr_N) == s && r != k) {
      pi[k] = pi_r_temp;
      i = 0;
      for (j = 0; j < n; j++) {
        if (phi1(j, distr_N) == t) {
          A[idx(i_loc(k, distr_M), j_loc(j, distr_N), nc)] = A_row_r_temp[i];
          ++i;
        }
      }
    }

    if (phi0(r, distr_N) == s && r != k) {
      pi[r] = pi_k_temp;
      i = 0;
      for (j = 0; j < n; j++) {
        if (phi1(j, distr_N) == t) {
          A[idx(i_loc(r, distr_M), j_loc(j, distr_N), nc)] = A_row_k_temp[i];
          ++i;
        }
      }
    }

    if (k == 29) {
      // printf("rank=%d s=%d t=%d", p_id, s, t);
      // print_array(nr, nc, A, distr_M, distr_N);

      // while (1)
      //   ;

      break;
    }

    // algo 2.4 begin

    // superstep 8
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);

    if (phi0(k, distr_M) == s && phi1(k, distr_N) == t) {
      // printf("the local a_kk before sending is: %f \n" , A[idx(i_loc(k,
      // distr_M), j_loc(k, distr_N), nc)]);
      for (i = 0; i < distr_M; ++i) {
        MPI_Send(&A[idx(i_loc(k, distr_M), j_loc(k, distr_N), nc)], 1,
                 MPI_DOUBLE, i * distr_M + t, k, MPI_COMM_WORLD);
      }
    }

    double a_kk;

    if (phi1(k, distr_N) == t) {
      for (i = 0; i < distr_N; ++i) {
        if (p_id == i * distr_M + t) {
          MPI_Recv(&a_kk, 1, MPI_DOUBLE, MPI_ANY_SOURCE, k, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
        }
      }
    }

    // superstep 9
    if (phi1(k, distr_N) == t) {
      for (i = k; i < n; ++i) {
        if (phi0(i, distr_M) == s) {
          if (fabs(a_kk) > EPS) {
            A[idx(i_loc(i, distr_M), j_loc(k, distr_N), nc)] /= a_kk;
          } else {
            printf(
                "rank %d: ABORT on k=%d because pivoting on zero element "
                "a_kk=%f\n ",
                p_id, k, a_kk);
            MPI_Abort(MPI_COMM_WORLD, 345);  // for some reason it aborts here
          }
        }
      }
    }

    // superstep 10

    // send every element of column k that P(s,t) owns, with row index
    // greater
    // than k,
    // to the processors P(s, *) in the same processor row
    if (phi1(k, distr_N) == t) {  // this processor owns the k-th column
      for (i = k; i < n; ++i) {
        if (phi0(i, distr_M) ==
            s) {  // this processor owns i-th element of kth column
          for (j = 0; j < distr_N; ++j) {
            assert(A[idx(i_loc(i, distr_M), j_loc(k, distr_N), nc)] < 10);
            MPI_Send(&A[idx(i_loc(i, distr_M), j_loc(k, distr_N), nc)], 1,
                     MPI_DOUBLE, s * distr_N + j, k, MPI_COMM_WORLD);
          }
        }
      }
    }

    double a_ik[n / distr_M];
    for (unsigned i = k; i < n; ++i) {
      if (phi0(i, distr_M) == s) {
        assert(i / distr_M < n / distr_M);
        MPI_Recv(&a_ik[i / distr_M], 1, MPI_DOUBLE,
                 s * distr_N + phi1(k, distr_N), k, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
    }

    if (phi0(k, distr_M) == s) {
      for (j = k; j < n; ++j) {
        if (phi1(j, distr_N) == t) {
          for (i = 0; i < distr_M; ++i) {
            assert(A[idx(i_loc(k, distr_M), j_loc(j, distr_N), nc)] < 10);
            MPI_Send(&A[idx(i_loc(k, distr_M), j_loc(j, distr_N), nc)], 1,
                     MPI_DOUBLE, i * distr_N + t, k, MPI_COMM_WORLD);
          }
        }
      }
    }

    double a_kj[n / distr_N];  // we will recieve one element per column
    for (unsigned j = k; j < n; ++j) {
      if (phi1(j, distr_N) == t) {
        assert(j / distr_N < n / distr_N);
        MPI_Recv(&a_kj[j / distr_N], 1, MPI_DOUBLE,
                 phi0(k, distr_M) * distr_N + t, k, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        assert(a_kj[j / distr_N] < 10);
      }
    }

    if (p_id == 0) {
      for (unsigned u = 0; u < n / distr_M; ++u) {
        printf("%f ", a_kj[u]);
      }

      printf("\n");
    }

    // superstep 11
    for (i = k; i < n; ++i) {
      if (phi0(i, distr_M) == s) {
        for (j = k; j < n; ++j) {
          if (phi1(j, distr_N) == t) {
            assert(i / distr_M < n / distr_M);
            assert(j / distr_N < n / distr_N);
            assert(a_ik[i / distr_M] < 10);
            // printf("a_kj[j / distr_N] == %f \n", a_kj[j / distr_N]);
            assert(a_kj[j / distr_N] < 10);
            A[idx(i_loc(i, distr_M), j_loc(j, distr_N), nc)] -=
                a_ik[i / distr_M] * a_kj[j / distr_N];
            // A[idx(i_loc(i, distr_M), j_loc(j, distr_N), nc)] -= 0;
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Retrieve problem size. */
  int n = N;

  unsigned distr_M,
      distr_N;  // M and N of the cyclic distr., need to be computed yet

  distr_M = largest_divisor(size);
  distr_N = size / distr_M;

  unsigned t = rank % distr_M;
  unsigned s = rank / distr_M;
  unsigned nr = (n + distr_M - s - 1) / distr_M;  // number of local rows
  unsigned nc = (n + distr_N - t - 1) / distr_N;  // number of local columns

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  double* A = (double*)malloc(nr * nc * sizeof(double));

  /* Initialize array(s). */
  init_array(n, nr, nc, distr_M, distr_N, A, s, t, rank);
  // if (rank == 0) print_array(nr, nc, A, distr_M, distr_N);

  /* Start timer. */
  polybench_start_instruments;

  unsigned* pi = malloc(sizeof(unsigned) * n);
  unsigned i;
  for (i = 0; i < n; ++i) {
    pi[i] = i;
  }

  /* Run kernel. */
  kernel_lu(n, A, rank, pi, distr_M, distr_N);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // if (rank == 0) polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  // /* Write results to file */
  MPI_Datatype cyclic_dist;
  int array_gsizes[2] = {n, n};
  int array_distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
  int array_dargs[2] = {1, 1};
  int array_psizes[2] = {distr_M, distr_N};

  MPI_Type_create_darray(size, rank, 2, array_gsizes, array_distribs,
                         array_dargs, array_psizes, MPI_ORDER_C, MPI_DOUBLE,
                         &cyclic_dist);

  MPI_Type_commit(&cyclic_dist);

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "lu.out", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native", MPI_INFO_NULL);
  MPI_File_write_at_all(file, 0, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  // if (rank == 2)
  //   MPI_File_write(file, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&file);

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  MPI_Finalize();

  return 0;
}
