#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void init_array(unsigned n, unsigned nr, unsigned nc, double* A) {
  unsigned i, j;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      A[i * nc + j] = rank;
    }
  }
}

static void print_array(int n, int nr, int nc, double* A) {
  int i, j;
  for (i = 0; i < nr; i++)
    for (j = 0; j < nc; j++) {
      if (j == 0) printf("\n");
      printf("%f ", A[i * nc + j]);
    }
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

int main(int argc, char** argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Retrieve problem size. */
  int n = 8;

  unsigned distr_M,
      distr_N;  // M and N of the cyclic distr., need to be computed yet

  distr_M = largest_divisor(size);
  distr_N = size / distr_M;

  unsigned s = rank % distr_M;
  unsigned t = rank / distr_M;
  unsigned nr = (n + distr_M - s - 1) / distr_M;  // number of local rows
  unsigned nc = (n + distr_N - t - 1) / distr_N;  // number of local columns

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  double* A = (double*)malloc(nr * nc * sizeof(double));

  /* Initialize array(s). */
  init_array(n, nr, nc, A);

  print_array(n, nr, nc, A);

  printf("n=%d, distr_M=%d, distr_N=%d\n", n, distr_M, distr_N);

  // /* Write results to file */
  MPI_Datatype cyclic_dist;
  int array_gsizes[2] = {n, n};
  int array_distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
  int array_dargs[2] = {2, 2};
  int array_psizes[2] = {distr_M, distr_N};

  MPI_Type_create_darray(size, rank, 2, array_gsizes, array_distribs,
                         array_dargs, array_psizes, MPI_ORDER_C, MPI_DOUBLE,
                         &cyclic_dist);

  MPI_Type_commit(&cyclic_dist);

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "lu.out", MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  MPI_File_set_view(file, 0, MPI_DOUBLE, cyclic_dist, "native", MPI_INFO_NULL);
  MPI_File_write_at_all(file, s * array_dargs[0] * n + array_dargs[1] * t, A,
                        nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  printf("%d\n", nr * nc);

  // if (rank == 2)
  //   MPI_File_write(file, A, nr * nc, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&file);

  printf("%f\n", A[0]);

  /* Be clean. */
  free(A);

  MPI_Finalize();

  return 0;
}
