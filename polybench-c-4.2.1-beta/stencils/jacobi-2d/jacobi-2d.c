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
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
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
   including the call and return. */
static void kernel_jacobi_2d(int tsteps, int nx, int ny,
                             DATA_TYPE POLYBENCH_2D(A, nx, ny, nx, nx),
                             DATA_TYPE POLYBENCH_2D(B, nx, ny, nx, ny)) {
  int t, i, j;

#pragma scop
  /* TODO: Exchange ghost cells */

  for (t = 0; t < _PB_TSTEPS; t++) {
    for (i = 1; i < nx - 1; i++)
      for (j = 1; j < ny - 1; j++)
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] +
                                     A[1 + i][j] + A[i - 1][j]);
    for (i = 1; i < nx - 1; i++)
      for (j = 1; j < ny - 1; j++)
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] +
                                     B[1 + i][j] + B[i - 1][j]);
  }
#pragma endscop
}

int main(int argc, char** argv) {
  /* Initialize MPI */
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);

  printf("%d = %d x %d\n", size, dims[0], dims[1]);

  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);

  /* Find neighbours of each rank */
  int rank_top, rank_bottom, rank_left, rank_right;
  MPI_Cart_shift(comm_cart, 0, 1, &rank_top, &rank_bottom);
  MPI_Cart_shift(comm_cart, 1, 1, &rank_left, &rank_right);

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Calculate problem size of local domain for every rank */
  int nx_local = n / dims[0];
  if (rank_right == MPI_PROC_NULL) nx_local += n % dims[0];

  int ny_local = n / dims[1];
  if (rank_bottom == MPI_PROC_NULL) ny_local += n % dims[1];

  /* TODO: Define a data type for sending columns of local domains */

  int sendcounts[size];
  int senddispls[size];
  MPI_Datatype blocktypes[dims[0] * dims[1]];
  int recvcounts[size];
  int recvdispls[size];
  MPI_Datatype recvtypes[size];

  for (int proc = 0; proc < size; proc++) {
    recvcounts[proc] = 0;
    recvdispls[proc] = nx_local + 3;
    recvtypes[proc] = MPI_DOUBLE;

    sendcounts[proc] = 0;
    senddispls[proc] = 0;
  }
  recvcounts[0] = nx_local * ny_local;

  if (rank == 0) {
    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    /* TODO: Define a data type for sending parts of the matrix back and forth
     */
    int subsizes[2];
    int starts[2] = {0, 0};
    const int globalsizes[2] = {n, n};

    /*initialize dimensions of the submatrices for the "interior" submatrices*/
    subsizes[0] = nx_local;
    subsizes[1] = ny_local;

    printf("%d x %d\n", subsizes[0], subsizes[1]);
    for (int i = 0; i < dims[0] - 1; i++) {
      for (int j = 0; j < dims[1] - 1; j++) {
        MPI_Type_create_subarray(2, globalsizes, subsizes, starts, MPI_ORDER_C,
                                 MPI_DOUBLE, &blocktypes[dims[0] * i + j]);
        MPI_Type_commit(&blocktypes[dims[0] * i + j]);
        printf("interior %d\n", dims[0] * i + j);
      }
    }

    /*initialize dimensions for the last row */
    subsizes[1] = ny_local + n - ny_local * dims[1];
    subsizes[0] = nx_local;
    for (int j = 0; j < dims[1] - 1; ++j) {
      MPI_Type_create_subarray(2, globalsizes, subsizes, starts, MPI_ORDER_C,
                               MPI_DOUBLE,
                               &blocktypes[dims[0] * (dims[1] - 1) + j]);
      MPI_Type_commit(&blocktypes[dims[0] * (dims[1] - 1) + j]);
      printf("last row %d\n", dims[0] * (dims[1] - 1) + j);
    }

    /* Initialize the dimensions for the last column */
    subsizes[0] = nx_local + n - nx_local * dims[0];
    subsizes[1] = ny_local;
    for (int i = 0; i < dims[0] - 1; ++i) {
      MPI_Type_create_subarray(2, globalsizes, subsizes, starts, MPI_ORDER_C,
                               MPI_DOUBLE, &blocktypes[dims[1] * (i + 1) - 1]);
      printf("last column %d\n", dims[0] * (i + 1) - 1);
      MPI_Type_commit(&blocktypes[dims[0] * (i + 1) - 1]);
    }

    /*initialize dimensions for the lower-right corner submatrix*/
    subsizes[0] = nx_local + n - nx_local * dims[0];
    subsizes[1] = ny_local + n - ny_local * dims[1];
    MPI_Type_create_subarray(2, globalsizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &blocktypes[dims[0] * dims[1] - 1]);
    printf("corner %d\n", dims[0] * dims[1] - 1);
    MPI_Type_commit(&blocktypes[dims[0] * dims[1] - 1]);

    /* now figure out the displacement and type of each processor's data */
    for (int proc = 0; proc < size; proc++) {
      int coords[2];

      MPI_Cart_coords(comm_cart, proc, 2, coords);

      sendcounts[proc] = 1;
      senddispls[proc] =
          (coords[0] * ny_local * n + coords[1] * nx_local) * sizeof(double);
    }

    printf("finished displ calc\n");

    POLYBENCH_2D_ARRAY_DECL(A_local, DATA_TYPE, nx_local + 2, ny_local + 2,
                            nx_local + 2, ny_local + 2);
    POLYBENCH_2D_ARRAY_DECL(B_local, DATA_TYPE, nx_local + 2, ny_local + 2,
                            nx_local + 2, ny_local + 2);

    printf("declared arrays\n");

    /* TODO: Send the segregated domain to all other ranks */
    MPI_Alltoallw(A, sendcounts, senddispls, blocktypes, A_local, recvcounts,
                  recvdispls, recvtypes, MPI_COMM_WORLD);

    printf("performed alltoall\n");

    /* Run kernel. */
    kernel_jacobi_2d(tsteps, nx_local, ny_local, POLYBENCH_ARRAY(A_local),
                     POLYBENCH_ARRAY(B_local));

    printf("finished kernel\n");

    /* Stop and print timer. */
    /* Start timer. */
    polybench_start_instruments;
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(A_local);
    POLYBENCH_FREE_ARRAY(B_local);
  } else {
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, nx_local + 2, ny_local + 2,
                            nx_local + 2, ny_local + 2);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, nx_local + 2, ny_local + 2,
                            nx_local + 2, ny_local + 2);

    MPI_Alltoallw(A, sendcounts, senddispls, blocktypes, A, recvcounts,
                  recvdispls, recvtypes, MPI_COMM_WORLD);

    /* Run kernel. */
    kernel_jacobi_2d(tsteps, nx_local, ny_local, POLYBENCH_ARRAY(A),
                     POLYBENCH_ARRAY(B));

    // /* TODO: Send local domain back to main rank */
    // MPI_Gatherv();
    // MPI_Gatherv();

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
  }

  MPI_Finalize();

  return 0;
}
