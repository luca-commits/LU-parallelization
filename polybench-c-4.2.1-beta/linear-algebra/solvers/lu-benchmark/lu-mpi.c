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

#ifdef HYBRID
#include <omp.h>
#endif

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
  // printf("rank=%d s=%d t=%d\n", p_id, s, t);

  for (unsigned i = 0; i < nr; ++i) {
    for (unsigned j = 0; j < nc; ++j) {
      if (j_glob(j, distr_N, t) < i_glob(i, distr_M, s)) {
        A[idx(i, j, nc)] = ((double)(-j_glob(j, distr_N, t) % n) / n + 1) * n;
      } else if (i_glob(i, distr_M, s) == j_glob(j, distr_N, t)) {
        A[idx(i, j, nc)] = 1 * n;
      } else {
        A[idx(i, j, nc)] = 0;
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
static void kernel_lu(int n, double* A, unsigned p_id, unsigned s, unsigned t,
                      unsigned* pi, const unsigned distr_M,
                      const unsigned distr_N, MPI_Comm comm_row,
                      MPI_Comm comm_col) {
  // unsigned nr = (n + distr_M - s - 1) / distr_M;  // number of local rows
  unsigned nr = (n + distr_M - s - 1) / distr_M;
  unsigned nc = (n + distr_N - t - 1) / distr_N;  // number of local columns
  int i, j, k, r;
  int start_i, start_j;
  DATA_TYPE absmax;

  int pi_k_temp, pi_r_temp;
  double A_row_k_temp[nc];
  double A_row_r_temp[nc];

  printf("pid: %d \n", p_id);
  // algorithm 2.5

  // find largest absolute value in column k
  double* Max = (double*)malloc(distr_M * sizeof(double));
  int* IMax = (int*)malloc(distr_M * sizeof(int));

  for (k = 0; k < n; k++) {
    MPI_Pcontrol(1, "Superstep (0)-(1)");
    if (phi1(k, distr_N) == t) {
      int rs;

      int ceil_n = n;
      if (n % distr_M != 0) ceil_n += distr_M - n % distr_M;

      if (ceil_n - k <= s) {
        int absmax_idx =
            (cblas_idamax(nr - i_loc(k, distr_M),
                          &A[idx(i_loc(k, distr_M), j_loc(k, distr_N), nc)],
                          nc)) +
            i_loc(k, distr_M);

        if (absmax_idx == i_loc(k, distr_M))
          rs = k;
        else
          rs = i_glob(absmax_idx, distr_M, s);

        absmax = fabs(A[idx(absmax_idx, j_loc(k, distr_N), nc)]);
      } else {
        absmax = 0;
        rs = 0;
      }

      double max = 0;
      if (absmax > EPS) {
        max = A[idx(i_loc(rs, distr_M), j_loc(k, distr_N), nc)];
      }

      MPI_Request requests[2];

      int recvcounts[distr_M];
      int recvdispls[distr_M];

      for (i = 0; i < distr_M; ++i) {
        recvcounts[i] = 1;
        recvdispls[i] = i;
      }

      MPI_Iallgatherv(&max, 1, MPI_DOUBLE, Max, recvcounts, recvdispls,
                      MPI_DOUBLE, comm_col, &requests[0]);
      MPI_Iallgatherv(&rs, 1, MPI_INT, IMax, recvcounts, recvdispls, MPI_INT,
                      comm_col, &requests[1]);

      MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    }
    MPI_Pcontrol(-1, "Superstep (0)-(1)");

    MPI_Pcontrol(1, "Superstep (2)");
    // superstep 2 & 3
    if (phi1(k, distr_N) == t) {
      absmax = 0;
      unsigned smax = 0;

      // for (i = 0; i < distr_M; i++) {
      //   if (fabs(Max[i]) > absmax) {
      //     absmax = fabs(Max[i]);
      //     smax = i;
      //   }
      // }

      smax = cblas_idamax(distr_M, Max, 1);
      absmax = fabs(Max[smax]);

      if (absmax > EPS) {
        int imax = IMax[smax];
        r = imax;  // global index
      } else {
        printf(
            "rank %d: ABORT because all elements in column are == 0 "
            "(absmax=%f)\n",
            p_id, absmax);
        MPI_Abort(MPI_COMM_WORLD, 192);
      }
    }
    MPI_Pcontrol(-1, "Superstep (2)");

    /* Superstep (3) */
    MPI_Pcontrol(1, "Superstep (3)");
    MPI_Bcast(&r, 1, MPI_INT, phi1(k, distr_N), comm_row);
    MPI_Pcontrol(-1, "Superstep (3)");

    /* Superstep (4)-(7) */
    MPI_Pcontrol(1, "Superstep (4)-(7)");
    if (phi0(k, distr_M) == s && phi0(r, distr_M) == s && r != k) {
      double a_temp;
      int pi_temp;

      // for (j = 0; j < nc; ++j) {  // waistful looping... will correct later
      //   a_temp = A[idx(i_loc(k, distr_M), j, nc)];
      //   A[idx(i_loc(k, distr_M), j, nc)] = A[idx(i_loc(r, distr_M), j, nc)];
      //   A[idx(i_loc(r, distr_M), j, nc)] = a_temp;
      // }

      cblas_dswap(nc, &A[idx(i_loc(k, distr_M), 0, nc)], 1,
                  &A[idx(i_loc(r, distr_M), 0, nc)], 1);

      pi_temp = pi[i_loc(r, distr_M)];
      pi[i_loc(r, distr_M)] = pi[i_loc(k, distr_M)];
      pi[i_loc(k, distr_M)] = pi_temp;

    } else if (phi0(k, distr_M) == s && r != k) {
      // Sendrecv here
      MPI_Sendrecv_replace(&A[idx(i_loc(k, distr_M), 0, nc)], nc, MPI_DOUBLE,
                           distr_N * phi0(r, distr_M) + t, 0,
                           distr_N * phi0(r, distr_M) + t, 1, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
      MPI_Sendrecv_replace(
          &pi[i_loc(k, distr_M)], 1, MPI_INT, distr_N * phi0(r, distr_M) + t, 2,
          distr_N * phi0(r, distr_M) + t, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (phi1(r, distr_M) == s && r != k) {
      // Sendrecv here
      MPI_Sendrecv_replace(&A[idx(i_loc(r, distr_M), 0, nc)], nc, MPI_DOUBLE,
                           distr_N * phi0(k, distr_M) + t, 1,
                           distr_N * phi0(k, distr_M) + t, 0, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
      MPI_Sendrecv_replace(
          &pi[i_loc(r, distr_M)], 1, MPI_INT, distr_N * phi0(k, distr_M) + t, 3,
          distr_N * phi0(k, distr_M) + t, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Pcontrol(-1, "Superstep (4)-(7)");

    // if (k == 1) break;

    // algo 2.4 begin

    // superstep 8
    MPI_Pcontrol(1, "Superstep (8)");
    double a_kk;

    if (phi1(k, distr_N) == t) {
      if (phi0(k, distr_M) == s) {
        a_kk = A[idx(i_loc(k, distr_M), j_loc(k, distr_N), nc)];
      }

      MPI_Bcast(&a_kk, 1, MPI_DOUBLE, phi0(k, distr_M), comm_col);
    }
    MPI_Pcontrol(-1, "Superstep (8)");

    // superstep 9
    MPI_Pcontrol(1, "Superstep (9)");
    if (phi1(k, distr_N) == t) {
      if ((k + 1) % distr_M < s + 1)
        start_i = i_loc(k + 1, distr_M);
      else
        start_i = i_loc(k + 1, distr_M) + 1;

      if (fabs(a_kk) < EPS) {
        printf(
            "rank %d: ABORT on k=%d because pivoting on zero element "
            "a_kk=%f\n ",
            p_id, k, a_kk);
        MPI_Abort(MPI_COMM_WORLD, 345);
      }

      // for (i = start_i; i < nr; ++i) {
      //   A[idx(i, j_loc(k, distr_N), nc)] /= a_kk;
      // }

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nr - start_i, 1, 1,
                  0, NULL, 1, NULL, 1, 1. / a_kk,
                  &A[idx(start_i, j_loc(k, distr_N), nc)], nc);
    }
    MPI_Pcontrol(-1, "Superstep (9)");

    // superstep 10
    MPI_Pcontrol(1, "Superstep (10)");
    double a_ik[nr];

    if (phi1(k, distr_N) == t) {
      // for (i = 0; i < nr; ++i) {
      //   a_ik[i] = A[idx(i, j_loc(k, distr_N), nc)];
      // }

      cblas_dcopy(nr, &A[idx(0, j_loc(k, distr_N), nc)], nc, a_ik, 1);
    }

    MPI_Bcast(a_ik, nr, MPI_DOUBLE, phi1(k, distr_N), comm_row);

    double a_kj[nc];

    if (phi0(k, distr_M) == s) {
      // for (j = 0; j < nc; ++j) {
      //   a_kj[j] = A[idx(i_loc(k, distr_M), j, nc)];
      // }

      cblas_dcopy(nc, &A[idx(i_loc(k, distr_M), 0, nc)], 1, a_kj, 1);
    }

    MPI_Bcast(a_kj, nc, MPI_DOUBLE, phi0(k, distr_M), comm_col);

    MPI_Pcontrol(-1, "Superstep (10)");

    // superstep 11
    MPI_Pcontrol(1, "Superstep (11)");
    int start_i, start_j;

    if ((k + 1) % distr_M < s + 1)
      start_i = i_loc(k + 1, distr_M);
    else
      start_i = i_loc(k + 1, distr_M) + 1;

    if ((k + 1) % distr_N < t + 1)
      start_j = j_loc(k + 1, distr_N);
    else
      start_j = j_loc(k + 1, distr_N) + 1;

    // for (i = start_i; i < nr; ++i) {
    //   for (j = start_j; j < nc; ++j) {
    //     A[idx(i, j, nc)] -= a_ik[i] * a_kj[j];
    //   }
    // }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nr - start_i,
                nc - start_j, 1, -1., &a_ik[start_i], 1, &a_kj[start_j], 1, 1.,
                &A[idx(start_i, start_j, nc)], nc);

    MPI_Pcontrol(-1, "Superstep (11)");

    // if (p_id == 0) printf("Finished step k=%d\n", k);
  }
}

int main(int argc, char** argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n, runs;

#ifdef HYBRID
  printf("Available OpenMP threads: %d\n", omp_get_max_threads());
  printf("Available MKL threads: %d\n", mkl_get_max_threads());
#endif

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

  unsigned distr_M,
      distr_N;  // M and N of the cyclic distr., need to be computed yet

  if (size != 1) {
    distr_M = largest_divisor(size);
    distr_N = size / distr_M;
  } else {
    distr_M = 1;
    distr_N = 1;
  }

  int dims[2] = {distr_M, distr_N};

  int periods[2] = {0, 1};
  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);

  int coords[2];
  MPI_Cart_coords(comm_cart, rank, 2, coords);
  unsigned s = coords[0];
  unsigned t = coords[1];

  printf("rank %d: s=%d t=%d\n", rank, s, t);

  MPI_Comm comm_row;
  int remain_dims_row[2] = {0, 1};
  MPI_Cart_sub(comm_cart, remain_dims_row, &comm_row);

  MPI_Comm comm_col;
  int remain_dims_col[2] = {1, 0};
  MPI_Cart_sub(comm_cart, remain_dims_col, &comm_col);

  unsigned nr = (n + distr_M - s - 1) / distr_M;  // number of local rows
  unsigned nc = (n + distr_N - t - 1) / distr_N;  // number of local columns

  int rank_row, rank_col;
  MPI_Comm_rank(comm_row, &rank_row);
  MPI_Comm_rank(comm_col, &rank_col);

  if (rank_row != t || rank_col != s) {
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  double* A = (double*)malloc(nr * nc * sizeof(double));
  int* pi = malloc(sizeof(int) * nr);

  double timings[runs];

  for (int i = 0; i < runs; ++i) {
    /* Initialize array(s). */
    // srand((rank + 1) * time(NULL));
    srand(rank * 10000);
    init_array(n, nr, nc, distr_M, distr_N, A, s, t, rank);
    // if (rank == 0) print_array(nr, nc, A, distr_M, distr_N);

    unsigned j;
    for (j = 0; j < nr; ++j) {
      pi[j] = i_glob(j, distr_M, s);
    }

#ifdef WRITE_TO_DISK
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
    MPI_File_open(MPI_COMM_WORLD, "lu_init.out",
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native",
                      MPI_INFO_NULL);
    MPI_File_write_at_all(file, 0, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
#endif

    /* Start timer. */
    double start_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    /* Run kernel. */
    MPI_Pcontrol(1, "Kernel");
    kernel_lu(n, A, rank, s, t, pi, distr_M, distr_N, comm_row, comm_col);
    MPI_Pcontrol(-1, "Kernel");

    /* Stop and print timer. */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) timings[i] = MPI_Wtime() - start_time;

      /* Prevent dead-code elimination. All live-out data must be printed
         by the function call in argument. */
      // if (rank == 0) polybench_prevent_dce(print_array(n,
      // POLYBENCH_ARRAY(A)));

#ifdef WRITE_TO_DISK
    // /* Write results to file */
    MPI_File_open(MPI_COMM_WORLD, "lu.out", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native",
                      MPI_INFO_NULL);
    MPI_File_write_at_all(file, 0, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&file);

    if (rank == 0) {
      unsigned* pi_full = (unsigned*)malloc(sizeof(unsigned) * n);

      for (i = 0; i < n; ++i) {
        if (phi0(i, distr_M) != s)
          MPI_Recv(&pi_full[i], 1, MPI_INT, phi0(i, distr_M) * distr_N, i,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
          pi_full[i] = pi[i_loc(i, distr_M)];
      }

      MPI_File file_pi;
      MPI_File_open(MPI_COMM_SELF, "pi.out", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                    MPI_INFO_NULL, &file_pi);

      MPI_File_write(file_pi, pi_full, n, MPI_INT, MPI_STATUS_IGNORE);

      MPI_File_close(&file_pi);
    } else if (t == 0) {
      for (i = 0; i < nr; ++i) {
        MPI_Send(&pi[i], 1, MPI_INT, 0, i_glob(i, distr_M, s), MPI_COMM_WORLD);
      }
    }
#endif

    if (rank == 0) printf("finished run %d\n", i);
  }

  if (rank == 0) {
#ifndef HYBRID
    int true_size = size;
#else
    int true_size = size * mkl_get_max_threads();
    printf("MKL max threads: %d\n", mkl_get_max_threads());
#endif

    char* filename;
    asprintf(&filename, "%d.csv", true_size);
    FILE* timings_file = fopen(filename, "w");

    fprintf(timings_file, "n,p,time,value\n");

    for (int i = 0; i < runs; ++i) {
      fprintf(timings_file, "%d,%d,%f,%f\n", n, true_size, timings[i], 0.0);
    }

    fclose(timings_file);
  }

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  // while (1)
  //   ;
  MPI_Finalize();

  return 0;
}
