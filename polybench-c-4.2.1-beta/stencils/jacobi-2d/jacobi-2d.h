/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _JACOBI_2D_H
# define _JACOBI_2D_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(TSTEPS) && !defined(N)
/* Define sample dataset sizes. */
/* Change TSTEPS/N from 20/30 to 1700/5000 */
#  ifdef MINI_DATASET
#   define TSTEPS 1700
#   define N 5000
#  endif

/* Change TSTEPS/N from 40/90 to 2000/6300 */
#  ifdef SMALL_DATASET
#   define TSTEPS 2000
#   define N 6300
#  endif

/* Change TSTEPS/N from 100/250 to 2500/7937 */
#  ifdef MEDIUM_DATASET
#   define TSTEPS 2500
#   define N 7937
#  endif

/* Change TSTEPS/N from 500/1300 to 3000/10000 */
#  ifdef LARGE_DATASET
#   define TSTEPS 3000
#   define N 10000
#  endif

/* Change TSTEPS/N from 1000/2800 to 3500/12599 */
#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 3500
#   define N 12599
#  endif


#endif /* !(TSTEPS N) */

# define _PB_TSTEPS POLYBENCH_LOOP_BOUND(TSTEPS,tsteps)
# define _PB_N POLYBENCH_LOOP_BOUND(N,n)


/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_DOUBLE
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !_JACOBI_2D_H */
