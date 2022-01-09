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
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef HYBRID
#include <omp.h>
#endif

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Matrix norm */
static double frobenius_norm(int nx, int ny, double* A) {
  double norm_sq = 0;

  for (int i = 0; i < nx * ny; i++) {
    norm_sq += A[i] * A[i];
  }

  return sqrt(norm_sq);
}

unsigned long get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000000 + tp.tv_usec;
}

/* Array initialization. */
static void init_array(int n, int nx, int ny, double* A, double* B, int s,
                       int t) {
  int i, j;

#pragma omp parallel for collapse(2)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      A[ny * i + j] = ((DATA_TYPE)(i * s) * (j * t + 2) + 2) / n;
      B[ny * i + j] = ((DATA_TYPE)(i * s) * (j * t + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, int ny, double* A)

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < ny; i++)
    for (j = 0; j < nx; j++) {
      if (j == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[nx * i + j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

static void exchange_cells(double* A, int nx_local, int ny_local,
                           int* neighbours, MPI_Request* requests,
                           MPI_Comm* comm_cart, MPI_Datatype* column_vec) {
  for (int i = 0; i < 8; i++) requests[i] = MPI_REQUEST_NULL;

  /* Communication with top neighbour */
  if (neighbours[0] != MPI_PROC_NULL) {
    MPI_Isend(&A[ny_local + 3], ny_local, MPI_DOUBLE, neighbours[0], 0,
              *comm_cart, &requests[0]);
    MPI_Irecv(&A[1], ny_local, MPI_DOUBLE, neighbours[0], MPI_ANY_TAG,
              *comm_cart, &requests[1]);
  } else {
    requests[0] = MPI_REQUEST_NULL;
    requests[1] = MPI_REQUEST_NULL;
  }

  /* Communication with bottom neighbour */
  if (neighbours[1] != MPI_PROC_NULL) {
    MPI_Isend(&A[(ny_local + 2) * (nx_local + 2) - 2 * (ny_local + 2) + 1],
              ny_local, MPI_DOUBLE, neighbours[1], 0, *comm_cart, &requests[2]);
    MPI_Irecv(&A[(ny_local + 2) * (nx_local + 2) - ny_local - 1], ny_local,
              MPI_DOUBLE, neighbours[1], MPI_ANY_TAG, *comm_cart, &requests[3]);
  } else {
    requests[2] = MPI_REQUEST_NULL;
    requests[3] = MPI_REQUEST_NULL;
  }

  /* Communication with left neighbour */
  if (neighbours[2] != MPI_PROC_NULL) {
    MPI_Isend(&A[ny_local + 3], 1, *column_vec, neighbours[2], 0, *comm_cart,
              &requests[4]);
    MPI_Irecv(&A[ny_local + 2], 1, *column_vec, neighbours[2], MPI_ANY_TAG,
              *comm_cart, &requests[5]);
  } else {
    requests[4] = MPI_REQUEST_NULL;
    requests[5] = MPI_REQUEST_NULL;
  }

  /* Communication with right neighbour */
  if (neighbours[3] != MPI_PROC_NULL) {
    MPI_Isend(&A[2 * (ny_local + 2) - 2], 1, *column_vec, neighbours[3], 0,
              *comm_cart, &requests[6]);
    MPI_Irecv(&A[2 * (ny_local + 2) - 1], 1, *column_vec, neighbours[3],
              MPI_ANY_TAG, *comm_cart, &requests[7]);
  } else {
    requests[6] = MPI_REQUEST_NULL;
    requests[7] = MPI_REQUEST_NULL;
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int tsteps, int nx_local, int ny_local, double* A,
                             double* B, int* neighbours, MPI_Comm* comm_cart,
                             MPI_Datatype* column_vec) {
  int t, i, j;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Request requests[8];
  MPI_Status statuses[8];

  /* Define bounds for iteration bounds, depending on location of processor
   * in the cartesian grid */
  int x_bound_low = 2;
  int x_bound_high = nx_local;
  int y_bound_low = 2;
  int y_bound_high = ny_local;

  for (t = 0; t < _PB_TSTEPS; t++) {
    exchange_cells(A, nx_local, ny_local, neighbours, requests, comm_cart,
                   column_vec);

// /* TODO: Implement SIMD instructions */
#ifdef HYBRID
#pragma omp parallel for collapse(2)
#endif
    for (i = x_bound_low; i < x_bound_high; i++)
      for (j = y_bound_low; j < y_bound_high; j++)
        B[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(ny_local + 2) * i + j] + A[(ny_local + 2) * i + j - 1] +
             A[(ny_local + 2) * i + 1 + j] + A[(ny_local + 2) * (1 + i) + j] +
             A[(ny_local + 2) * (i - 1) + j]);

    /* Wait for all communication to finish */
    int err = MPI_Waitall(8, requests, statuses);

    if (neighbours[0] != MPI_PROC_NULL) {
      j = 1;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (i = x_bound_low; i < x_bound_high; i++) {
        B[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(ny_local + 2) * i + j] + A[(ny_local + 2) * i + j - 1] +
             A[(ny_local + 2) * i + 1 + j] + A[(ny_local + 2) * (1 + i) + j] +
             A[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[1] != MPI_PROC_NULL) {
      j = ny_local;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (i = x_bound_low; i < x_bound_high; i++) {
        B[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(ny_local + 2) * i + j] + A[(ny_local + 2) * i + j - 1] +
             A[(ny_local + 2) * i + 1 + j] + A[(ny_local + 2) * (1 + i) + j] +
             A[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[2] != MPI_PROC_NULL) {
      i = 1;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (j = x_bound_low; j < x_bound_high; j++) {
        B[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(ny_local + 2) * i + j] + A[(ny_local + 2) * i + j - 1] +
             A[(ny_local + 2) * i + 1 + j] + A[(ny_local + 2) * (1 + i) + j] +
             A[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[3] != MPI_PROC_NULL) {
      i = nx_local;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (j = y_bound_low; j < y_bound_high; j++) {
        B[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (A[(ny_local + 2) * i + j] + A[(ny_local + 2) * i + j - 1] +
             A[(ny_local + 2) * i + 1 + j] + A[(ny_local + 2) * (1 + i) + j] +
             A[(ny_local + 2) * (i - 1) + j]);
      }
    }
    exchange_cells(B, nx_local, ny_local, neighbours, requests, comm_cart,
                   column_vec);

#ifdef HYBRID
#pragma omp parallel for collapse(2)
#endif
    for (i = x_bound_low; i < x_bound_high; i++)
      for (j = y_bound_low; j < y_bound_high; j++)
        A[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(ny_local + 2) * i + j] + B[(ny_local + 2) * i + j - 1] +
             B[(ny_local + 2) * i + 1 + j] + B[(ny_local + 2) * (1 + i) + j] +
             B[(ny_local + 2) * (i - 1) + j]);

    /* Wait for all communication to finish */
    err = MPI_Waitall(8, requests, statuses);

    if (neighbours[0] != MPI_PROC_NULL) {
      j = 1;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (i = x_bound_low; i < x_bound_high; i++) {
        A[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(ny_local + 2) * i + j] + B[(ny_local + 2) * i + j - 1] +
             B[(ny_local + 2) * i + 1 + j] + B[(ny_local + 2) * (1 + i) + j] +
             B[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[1] != MPI_PROC_NULL) {
      j = ny_local;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (i = x_bound_low; i < x_bound_high; i++) {
        A[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(ny_local + 2) * i + j] + B[(ny_local + 2) * i + j - 1] +
             B[(ny_local + 2) * i + 1 + j] + B[(ny_local + 2) * (1 + i) + j] +
             B[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[2] != MPI_PROC_NULL) {
      i = 1;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (j = y_bound_low; j < y_bound_high; j++) {
        A[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(ny_local + 2) * i + j] + B[(ny_local + 2) * i + j - 1] +
             B[(ny_local + 2) * i + 1 + j] + B[(ny_local + 2) * (1 + i) + j] +
             B[(ny_local + 2) * (i - 1) + j]);
      }
    }

    if (neighbours[3] != MPI_PROC_NULL) {
      i = ny_local;

#ifdef HYBRID
#pragma omp parallel for
#endif
      for (j = y_bound_low; j < y_bound_high; j++) {
        A[(ny_local + 2) * i + j] =
            SCALAR_VAL(0.2) *
            (B[(ny_local + 2) * i + j] + B[(ny_local + 2) * i + j - 1] +
             B[(ny_local + 2) * i + 1 + j] + B[(ny_local + 2) * (1 + i) + j] +
             B[(ny_local + 2) * (i - 1) + j]);
      }
    }
  }
}

int main(int argc, char** argv) {
  /* Initialize MPI */
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  int dims_temp[2] = {0, 0};
  int periods[2] = {0, 0};
  MPI_Dims_create(size, 2, dims_temp);

  int dims[2] = {dims_temp[0],
                 dims_temp[1]};  // definition: dims[0] as no of ranks in
                                 // "x-direction" (to the right), dims[1] as no
                                 // of ranks in "y-direction" (to the bottom)

  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims_temp, periods, 0, &comm_cart);

  // MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int coords[2];
  MPI_Cart_coords(comm_cart, rank, 2, coords);

  /* Find neighbours of each rank */
  int neighbours[4];  // top, bottom, left, right
  MPI_Cart_shift(comm_cart, 0, 1, &neighbours[0], &neighbours[1]);
  MPI_Cart_shift(comm_cart, 1, 1, &neighbours[2], &neighbours[3]);

  // printf("Rank %d: %d %d %d %d\n", rank, neighbours[0], neighbours[1],
  //        neighbours[2], neighbours[3]);

  /* Calculate problem size of local domain for every rank */
  int nx_local = n / dims[0];
  if (neighbours[1] == MPI_PROC_NULL) nx_local += n % dims[0];

  int ny_local = n / dims[1];
  if (neighbours[3] == MPI_PROC_NULL) ny_local += n % dims[1];

  printf("rank %d: %d x %d\n", rank, nx_local, ny_local);

  /* Define a data type for sending columns of local domains */
  MPI_Datatype column_vec;
  MPI_Type_vector(nx_local, 1, ny_local + 2, MPI_DOUBLE, &column_vec);
  MPI_Type_commit(&column_vec);

  double* A_local =
      (double*)malloc((nx_local + 2) * (ny_local + 2) * sizeof(double));
  double* B_local =
      (double*)malloc((nx_local + 2) * (ny_local + 2) * sizeof(double));

  // printf("malloced\n");

  /* Initialize array(s). */
  init_array(n, nx_local, ny_local, A_local, B_local, coords[0], coords[1]);

  // printf("finished init\n");

  // if (rank == 1) {
  //   print_array(nx_local + 2, ny_local + 2, A_local);
  //   print_array(nx_local + 2, ny_local + 2, B_local);
  // }

  double timings[runs];
  double start_time;

  for (int i = 0; i < runs; ++i) {
    MPI_Pcontrol(1, "Kernel");

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    /* Run kernel. */
    kernel_jacobi_2d(tsteps, nx_local, ny_local, A_local, B_local, neighbours,
                     &comm_cart, &column_vec);

    /* Stop and print timer. */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) timings[i] = MPI_Wtime() - start_time;
    MPI_Pcontrol(-1, "Kernel");
  }

  if (rank == 0) {
#ifndef HYBRID
    int true_size = size;
#else
    int true_size = size * omp_get_max_threads();
#endif

    char* filename;
    asprintf(&filename, "%d.csv", true_size);
    FILE* timings_file = fopen(filename, "w");

    fprintf(timings_file, "n,tsteps,p,time,value\n");

    for (int i = 0; i < runs; ++i) {
      fprintf(timings_file, "%d,%d,%d,%f,%f\n", n, tsteps, true_size,
              timings[i], 0.0);
    }

    fclose(timings_file);
  }

  /* Be clean. */
  free(A_local);
  free(B_local);

  MPI_Finalize();

  return 0;
}