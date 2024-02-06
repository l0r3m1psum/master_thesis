// cc blas.c -framework Accelerate

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include "blas.h"
#undef ACCELERATE_NEW_LAPACK
#undef ACCELERATE_LAPACK_ILP64

#define LEFT_SQUARE_BRACKET               "["
#define RIGHT_SQUARE_BRACKET              "]"
#define LEFT_SQUARE_BRACKET_UPPER_CORNER  "\u23A1"
#define LEFT_SQUARE_BRACKET_EXTENSION     "\u23A2"
#define LEFT_SQUARE_BRACKET_LOWER_CORNER  "\u23A3"
#define RIGHT_SQUARE_BRACKET_UPPER_CORNER "\u23A4"
#define RIGHT_SQUARE_BRACKET_EXTENSION    "\u23A5"
#define RIGHT_SQUARE_BRACKET_LOWER_CORNER "\u23A6"

void cblas_sprint(const enum CBLAS_ORDER ORDER,
				  const enum CBLAS_TRANSPOSE TRANS,
				  const long M,
				  const long N,
				  const float *A,
				  const long LD) {
	char rout[sizeof __func__] = {0};
	if ((ORDER != CblasRowMajor) & (ORDER != CblasColMajor)) {
		cblas_xerbla(1, strncpy(rout, __func__, sizeof rout),
					 "illegal ORDER setting: %d", ORDER);
	}
	if (ORDER != CblasRowMajor) {
		cblas_xerbla(1, strncpy(rout, __func__, sizeof rout),
					 "row-major is the only supported order for now");
	}
	if ((TRANS != CblasNoTrans) & (TRANS != CblasTrans)) {
		cblas_xerbla(2, strncpy(rout, __func__, sizeof rout),
					 "illegal TRANS setting: %d", TRANS);
	}
	if (TRANS != CblasNoTrans) {
		cblas_xerbla(2, strncpy(rout, __func__, sizeof rout),
					 "transposition is not supported for now");
	}
	if (M < 0) {
		cblas_xerbla(3, strncpy(rout, __func__, sizeof rout),
					 "the number of rows M=%l shoud be >= 1", M);
	}
	if (N == 0 && M != 0) {
		cblas_xerbla(3, strncpy(rout, __func__, sizeof rout),
					 "M=%l should be 0 since N=0", M);
	}
	if (N < 0) {
		cblas_xerbla(4, strncpy(rout, __func__, sizeof rout),
					 "the number of columns N=%l shoud be >= 1", N);
	}
	if (M == 0 && N != 0) {
		cblas_xerbla(4, strncpy(rout, __func__, sizeof rout),
					 "N=%l should be 0 since M=0", N);
	}
	if (M > LONG_MAX/(N == 0 ? 1 : N)) {
		cblas_xerbla(4, strncpy(rout, __func__, sizeof rout),
					 "M=%l*N=%l is bigger than LONG_MAX", M, N);
	}
	if (!A) {
		cblas_xerbla(5, strncpy(rout, __func__, sizeof rout),
					 "the pointer to the matrix should not be null");
	}
	// TODO: check if given the number of rows LD fits in MxN
	if (LD != N) {
		cblas_xerbla(6, strncpy(rout, __func__, sizeof rout),
					 "leading dimension not supported for now");
	}
	// TODO: add support for CBLAS_ORDER, CBLAS_TRANSPOSE, LD
	// TODO: do something smart for number formatting or accept format as an argument (like numpy).
	// By something smart I mean deciding a satisfactory precision to print all
	// the elements in the matrix (I wonder how numpy does it...)
	// https://www.ryanjuckett.com/printing-floating-point-numbers/
	// https://numpy.org/doc/stable/reference/generated/numpy.array2string.html

	// If printf fails we don't report any error since first blas does not have
	// a standard API to communicate this kinds of events and second this is a
	// best effort function.
	if (M == 0) {
		printf(LEFT_SQUARE_BRACKET RIGHT_SQUARE_BRACKET "\n");
		return;
	}
	for (long i = 0; i < M; i++) {
		char *open_square_bracket =
			(M == 1) ? LEFT_SQUARE_BRACKET
			: (i == 0) ? LEFT_SQUARE_BRACKET_UPPER_CORNER
			: (i < M-1) ? LEFT_SQUARE_BRACKET_EXTENSION
			: LEFT_SQUARE_BRACKET_LOWER_CORNER;
		printf("%s", open_square_bracket);
		long j = 0;
		for (; j < N-1; j++) {
			printf("%6.5f ", A[i*LD + j]);
		}
		char *close_square_bracket =
			(M == 1) ? RIGHT_SQUARE_BRACKET
			: (i == 0) ? RIGHT_SQUARE_BRACKET_UPPER_CORNER
			: (i < M-1) ? RIGHT_SQUARE_BRACKET_EXTENSION
			: RIGHT_SQUARE_BRACKET_LOWER_CORNER;
		assert(j == N-1);
		printf("%6.5f%s\n", A[i*LD + j], close_square_bracket);
	}
}

#ifdef TEST_blas
int main(void) {
	enum CBLAS_ORDER ORDER = CblasRowMajor;
	enum CBLAS_TRANSPOSE TRANSA = CblasNoTrans, TRANSB = CblasNoTrans;
	{
		__LAPACK_int M = 2, N = 2, K = 2;
		// MxK * KxN = MxN
		float ALPHA = 1, A[] = (float[]){1,2,3,4};
		__LAPACK_int LDA = K;
		float B[] = (float[]){1,2,3,4};
		__LAPACK_int LDB = N;
		float BETA = 0, C[] = (float[]){0,0,0,0};
		__LAPACK_int LDC = N;
		cblas_sgemm(
			ORDER, TRANSA, TRANSB,
			M, N, K,
			ALPHA, A, LDA,
			B, LDB,
			BETA, C, LDC
		);
		cblas_sprint(ORDER, TRANSA, M, N, C, LDC);
	}
	{
		// Outer product test
		__LAPACK_int M = 2, N = 2, K = 1;
		// MxK * KxN = MxN
		float ALPHA = 1, A[] = (float[]){1,2};
		__LAPACK_int LDA = K;
		float B[] = (float[]){1,2};
		__LAPACK_int LDB = N;
		float BETA = 0, C[] = (float[]){0,0,0,0};
		__LAPACK_int LDC = N;
		cblas_sgemm(
			ORDER, TRANSA, TRANSB,
			M, N, K,
			ALPHA, A, LDA,
			B, LDB,
			BETA, C, LDC
		);
		cblas_sprint(ORDER, TRANSA, M, N, C, LDC);
	}
	{
		// Inner product test
		__LAPACK_int M = 1, N = 1, K = 2;
		// MxK * K*N = M*N
		float ALPHA = 1, A[] = (float[]){1,2};
		__LAPACK_int LDA = K;
		float B[] = (float[]){1,2};
		__LAPACK_int LDB = N;
		float BETA = 0, C[] = (float[]){0,0,0,0};
		__LAPACK_int LDC = N;
		cblas_sgemm(
			ORDER, TRANSA, TRANSB,
			M, N, K,
			ALPHA, A, LDA,
			B, LDB,
			BETA, C, LDC
		);
		cblas_sprint(ORDER, TRANSA, M, N, C, LDC);
	}
	
	printf("\n");
	
	{
		__LAPACK_int N = 4;
		float ALPHA = 1;
		float X[] = (float[]){1,2,3,4};
		__LAPACK_int INCX = 1;
		float Y[] = (float[]){4,3,2,1};
		__LAPACK_int INCY = 1;
		cblas_saxpy(N, ALPHA, X, INCX, Y, INCY);
		cblas_sprint(ORDER, TRANSA, N, 1, Y, 1);
	}
	
	{
		// Zeroing a vector
		__LAPACK_int N = 4;
		float SX[] = (float[]){0};
		__LAPACK_int INCX = 0;
		float SY[] = (float[]){1,2,3,4};
		__LAPACK_int INCY = 1;
		cblas_scopy(N, SX, INCX, SY, INCY);
		cblas_sprint(ORDER, TRANSA, N, 1, SY, 1);
	}
	
	{
		__LAPACK_int N = 4;
		float SA = 4;
		float SX[] = (float[]){1,1,1,1};
		__LAPACK_int INCX = 1;
		cblas_sscal(N, SA, SX, INCX);
		cblas_sprint(ORDER, TRANSA, N, 1, SX, 1);
	}

	{
		__LAPACK_int N = 0;
		float SA = 4;
		float SX[] = (float[]){1,1,1,1};
		__LAPACK_int INCX = 1;
		cblas_sscal(N, SA, SX, INCX);
		cblas_sprint(ORDER, TRANSA, N, 0, SX, 0);
	}

	// cblas_sswap, cblas_sdot, cblas_sdsdot, cblas_snrm2, cblas_sasum, cblas_isamax, cblas_sgemv;
	return 0;
}
#endif
