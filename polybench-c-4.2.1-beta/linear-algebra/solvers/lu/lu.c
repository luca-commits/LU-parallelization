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
static void print_array(int nr, int nc, double *A, unsigned distr_M,
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
                       unsigned distr_N, double *A, unsigned s, unsigned t,
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
static void kernel_lu(int n, double *A, unsigned p_id, unsigned s, unsigned t,
                      unsigned *pi, const unsigned distr_M,
                      const unsigned distr_N, MPI_Comm comm_row,
                      MPI_Comm comm_col) {}

int main(int argc, char **argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Retrieve problem size. */
  int n = N;

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

  /* Initialize array(s). */
  // srand((rank + 1) * time(NULL));
  srand(rank * 10000);
  init_array(n, nr, nc, distr_M, distr_N, A, s, t, rank);
  // if (rank == 0) print_array(nr, nc, A, distr_M, distr_N);

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

  MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native", MPI_INFO_NULL);
  MPI_File_write_at_all(file, 0, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&file);
#endif

  int *pi = malloc(sizeof(int) * n);

  int desc[9];
  int info;
  descinit_(desc, &n, &n, &one, &one, &zero, &zero, &ictxt, &nc, &info);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  MPI_Pcontrol(1, "Kernel");
  pdgetrf_(&n, &n, A, &one, &one, desc, pi, &info);
  // char uplo = 'L';
  // pdpotrf_(&uplo, &n, A, &one, &one, desc, &info);
  MPI_Pcontrol(-1, "Kernel");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // if (rank == 0) polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

#ifdef WRITE_TO_DISK
  // /* Write results to file */
  MPI_File_open(MPI_COMM_WORLD, "lu.out", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native", MPI_INFO_NULL);
  MPI_File_write_at_all(file, 0, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&file);

  if (rank == 0) {
    unsigned *pi_full = (unsigned *)malloc(sizeof(unsigned) * n);

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
  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  blacs_gridexit_(&ictxt);
  MPI_Finalize();

  return 0;
}
