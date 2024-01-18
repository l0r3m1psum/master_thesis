#ifndef blas_h
#define blas_h

// https://www.netlib.org/blas/
// https://developer.apple.com/documentation/accelerate/blas?language=objc
// https://www.gnu.org/software/gsl/doc/html/blas.html
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/blas-functionality.html

// Good LDA explenations:
//   * https://www.cse-lab.ethz.ch/wp-content/uploads/2020/10/Linear-Algebra-BLAS-ISA-Pipelining.pdf
//   * https://stackoverflow.com/a/37891808
//   * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-001.html
// Honestly Intel has the best BLAS documentation.

#include <Accelerate/Accelerate.h>
void cblas_sprint(const enum CBLAS_ORDER ORDER,
				  const enum CBLAS_TRANSPOSE TRANS,
				  const __LAPACK_int M,
				  const __LAPACK_int N,
				  const float *A,
				  const __LAPACK_int LD);

#endif
