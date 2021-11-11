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
static void init_array(int n, double* A, double* B) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[n * i + j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[n * i + j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double* A)

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if (j == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[n * i + j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

static void exchange_cells(double* A, int nx_local, int ny_local,
                           int* neighbours, MPI_Comm* comm_cart,
                           MPI_Datatype* column_vec) {
  MPI_Request requests[8];
  for (int i = 0; i < 8; i++) requests[i] = MPI_REQUEST_NULL;

  /* Communication with top neighbour */
  if (neighbours[0] != MPI_PROC_NULL) {
    MPI_Isend(&A[nx_local + 3], nx_local, MPI_DOUBLE, neighbours[0], 0,
              *comm_cart, &requests[0]);
    MPI_Irecv(&A[1], nx_local, MPI_DOUBLE, neighbours[0], MPI_ANY_TAG,
              *comm_cart, &requests[1]);
  } else {
    requests[0] = MPI_REQUEST_NULL;
    requests[1] = MPI_REQUEST_NULL;
  }

  /* Communication with bottom neighbour */
  if (neighbours[1] != MPI_PROC_NULL) {
    MPI_Isend(&A[(nx_local + 2) * (ny_local + 2) - 2 * (nx_local + 2) + 1],
              nx_local, MPI_DOUBLE, neighbours[1], 0, *comm_cart, &requests[2]);
    MPI_Irecv(&A[(nx_local + 2) * (ny_local + 2) - nx_local - 1], nx_local,
              MPI_DOUBLE, neighbours[1], MPI_ANY_TAG, *comm_cart, &requests[3]);
  } else {
    requests[2] = MPI_REQUEST_NULL;
    requests[3] = MPI_REQUEST_NULL;
  }

  /* Communication with left neighbour */
  if (neighbours[2] != MPI_PROC_NULL) {
    MPI_Isend(&A[nx_local + 3], 1, *column_vec, neighbours[2], 0, *comm_cart,
              &requests[4]);
    MPI_Irecv(&A[nx_local + 2], 1, *column_vec, neighbours[2], MPI_ANY_TAG,
              *comm_cart, &requests[5]);
  } else {
    requests[4] = MPI_REQUEST_NULL;
    requests[5] = MPI_REQUEST_NULL;
  }

  /* Communication with right neighbour */
  if (neighbours[3] != MPI_PROC_NULL) {
    MPI_Isend(&A[2 * (nx_local + 2) - 2], 1, *column_vec, neighbours[3], 0,
              *comm_cart, &requests[6]);
    MPI_Irecv(&A[2 * (nx_local + 2) - 1], 1, *column_vec, neighbours[3],
              MPI_ANY_TAG, *comm_cart, &requests[7]);
  } else {
    requests[6] = MPI_REQUEST_NULL;
    requests[7] = MPI_REQUEST_NULL;
  }

  /* Wait for all communication to finish */
  MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int tsteps, int nx_local, int ny_local, double* A,
                             double* B, int* neighbours, MPI_Comm* comm_cart,
                             MPI_Datatype* column_vec) {
  int t, i, j;

  /* Define bounds for iteration bounds, depending on location of processor in
   * the cartesian grid */
  int x_bound_low = 1;
  int x_bound_high = nx_local + 1;
  int y_bound_low = 1;
  int y_bound_high = ny_local + 1;

  if (neighbours[0] == MPI_PROC_NULL) y_bound_low += 1;
  if (neighbours[1] == MPI_PROC_NULL) y_bound_high -= 1;
  if (neighbours[2] == MPI_PROC_NULL) x_bound_low += 1;
  if (neighbours[3] == MPI_PROC_NULL) x_bound_high -= 1;

  for (t = 0; t < _PB_TSTEPS; t++) {
    exchange_cells(A, nx_local, ny_local, neighbours, comm_cart, column_vec);
    /* TODO: Implement SIMD instructions */
    for (i = y_bound_low; i < y_bound_high; i++)
      for (j = x_bound_low; j < x_bound_high; j++)
        B[(nx_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(nx_local + 2) * i + j] + A[(nx_local + 2) * i + j - 1] +
             A[(nx_local + 2) * i + 1 + j] + A[(nx_local + 2) * (1 + i) + j] +
             A[(nx_local + 2) * (i - 1) + j]);

    exchange_cells(B, nx_local, ny_local, neighbours, comm_cart, column_vec);

    for (i = y_bound_low; i < y_bound_high; i++)
      for (j = x_bound_low; j < x_bound_high; j++)
        A[(nx_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(nx_local + 2) * i + j] + B[(nx_local + 2) * i + j - 1] +
             B[(nx_local + 2) * i + 1 + j] + B[(nx_local + 2) * (1 + i) + j] +
             B[(nx_local + 2) * (i - 1) + j]);
  }
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
  int neighbours[4];  // top, bottom, left, right
  MPI_Cart_shift(comm_cart, 0, 1, &neighbours[0], &neighbours[1]);
  MPI_Cart_shift(comm_cart, 1, 1, &neighbours[2], &neighbours[3]);

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Calculate problem size of local domain for every rank */
  int nx_local = n / dims[0];
  if (neighbours[3] == MPI_PROC_NULL) nx_local += n % dims[0];

  int ny_local = n / dims[1];
  if (neighbours[1] == MPI_PROC_NULL) ny_local += n % dims[1];

  /* Define a data type for sending columns of local domains */
  MPI_Datatype column_vec;
  MPI_Type_vector(ny_local, 1, nx_local + 2, MPI_DOUBLE, &column_vec);
  MPI_Type_commit(&column_vec);

  int sendcounts[size];
  int senddispls[size];
  MPI_Datatype blocktypes[size];
  int recvcounts[size];
  int recvdispls[size];
  MPI_Datatype recvtypes[size];

  /* Define a data type for each rank's subdomain */
  MPI_Datatype blocktype;
  MPI_Type_vector(ny_local, nx_local, nx_local + 2, MPI_DOUBLE, &blocktype);
  MPI_Type_commit(&blocktype);

  for (int proc = 0; proc < size; proc++) {
    recvcounts[proc] = 0;
    recvdispls[proc] = (nx_local + 3) * sizeof(double);
    recvtypes[proc] = blocktype;

    sendcounts[proc] = 0;
    senddispls[proc] = (nx_local + 3) * sizeof(double);
  }
  recvcounts[0] = 1;

  double* A;
  double* B;

  if (rank == 0) {
    /* Variable declaration/allocation. */
    // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    // POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

    A = (double*)malloc(n * n * sizeof(double));
    B = (double*)malloc(n * n * sizeof(double));

    /* Initialize array(s). */
    init_array(n, A, B);

    // print_array(n, A);

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
  }

  double* A_local =
      (double*)malloc((nx_local + 2) * (ny_local + 2) * sizeof(double));
  double* B_local =
      (double*)malloc((nx_local + 2) * (ny_local + 2) * sizeof(double));

  /* Send the segregated domain to all other ranks */
  MPI_Alltoallw(A, sendcounts, senddispls, blocktypes, A_local, recvcounts,
                recvdispls, recvtypes, MPI_COMM_WORLD);
  MPI_Alltoallw(B, sendcounts, senddispls, blocktypes, B_local, recvcounts,
                recvdispls, recvtypes, MPI_COMM_WORLD);

  // if (rank == 2) {
  //   print_array(nx_local + 2, A_local);
  //   print_array(nx_local + 2, B_local);
  // }

  if (rank == 0) {
    /* Start timer. */
    polybench_prepare_instruments();
    polybench_timer_start();
  }

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, nx_local, ny_local, A_local, B_local, neighbours,
                   &comm_cart, &column_vec);

  if (rank == 0) {
    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
  }

  // if (rank == 2) {
  //   print_array(nx_local + 2, A_local);
  //   print_array(nx_local + 2, B_local);
  // }

  /* TODO: Put submatrices back together */
  MPI_Alltoallw(A_local, recvcounts, recvdispls, recvtypes, A, sendcounts,
                senddispls, blocktypes, comm_cart);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  // if (rank == 0) print_array(n, A);

  /* Be clean. */
  if (rank == 0) {
    free(A);
    free(B);
  }

  free(A_local);
  free(B_local);

  MPI_Finalize();

  return 0;
}
