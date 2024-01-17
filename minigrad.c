#define IFF(a, b) (!!(a) == !!(b))
#define ARRAY_LEN(a) (sizeof (a) / sizeof *(a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#pragma mark BLAS (Basic Linear Algebra Subroutine)

// https://www.netlib.org/blas/
// https://developer.apple.com/documentation/accelerate/blas?language=objc
// https://www.gnu.org/software/gsl/doc/html/blas.html
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/blas-functionality.html

// Good LDA explenations:
//   * https://www.cse-lab.ethz.ch/wp-content/uploads/2020/10/Linear-Algebra-BLAS-ISA-Pipelining.pdf
//   * https://stackoverflow.com/a/37891808
//   * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-001.html
// Honestly Intel has the best BLAS documentation.

#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
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

// NOTE: should we return some error from printf?
void cblas_sprint(const enum CBLAS_ORDER ORDER,
				  const enum CBLAS_TRANSPOSE TRANS,
				  const __LAPACK_int M,
				  const __LAPACK_int N,
				  const float *A,
				  const __LAPACK_int LD) {
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
	if (M < 1) {
		cblas_xerbla(3, strncpy(rout, __func__, sizeof rout),
					 "the number of rows M=%d shoud be >= 1", M);
	}
	if (N < 1) {
		cblas_xerbla(4, strncpy(rout, __func__, sizeof rout),
					 "the number of columns N=%d shoud be >= 1", N);
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
	// TODO: check that M*N fits in an int (M <= INT_MAX/N)
	// NOTE: does BLAS handles empty matrices?
	// TODO: do something smart for number formatting or accept format as an argument.

	for (int i = 0; i < M; i++) {
		char *open_square_bracket =
			(M == 1) ? LEFT_SQUARE_BRACKET
			: (i == 0) ? LEFT_SQUARE_BRACKET_UPPER_CORNER
			: (i < M-1) ? LEFT_SQUARE_BRACKET_EXTENSION
			: LEFT_SQUARE_BRACKET_LOWER_CORNER;
		printf("%s", open_square_bracket);
		int j = 0;
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

#pragma mark Metal

// TODO: put Metal API stuff here.

#pragma mark Memory Allocator

#include <stdbool.h>
#include <stddef.h>
#include <assert.h>

typedef struct Arena {
	void  *data;
	size_t size;
	size_t used;
} Arena;

static bool
Arena_invariant(const Arena alloc[static 1]) {
	assert(alloc != NULL);
	if (alloc->data == NULL) {
		return (alloc->size == 0) & (alloc->used == 0);
	}
	return alloc->used <= alloc->size;
}

// TODO: add allignment capabilities for Metal.
static void *
Arena_alloc(Arena alloc[static 1], size_t req_size) {
	assert(Arena_invariant(alloc));
	if (alloc->size - alloc->used < req_size) {
		return NULL;
	}
	void *res = alloc->data + alloc->used;
	alloc->used += req_size;
	return res;
}

static void
Arena_reset(Arena alloc[static 1]) {
	assert(Arena_invariant(alloc));
	alloc->used = 0;
	assert(Arena_invariant(alloc));
}

#pragma mark Automatic Differentiation

typedef enum Operation {
	// 0-arity
	Operation_nop, // matrix
	// 1-arity
	Operation_sum, // reduction
	Operation_sig, // sigmoid
	Operation_neg,  // negation
	// 2-arity
	Operation_add, // element-wise (with broadcasting)
	Operation_mul, // element-wise (with broadcasting)
	Operation_pow, // element-wise (only by scalar)
	Operation_dot, // matrix multiplication (without vector promotion)
} Operation;

#define Operation_IS_BINARY(op) ((op) >= Operation_add)

// NOTE: probably passing tensor by value is even better due to the way in which the calling convention works.
typedef struct Tensor {
	float *value, *grad; // always the same shape.
	unsigned rows, cols;
	// recipe
	Operation op;
	unsigned arg0; // index in tape
	unsigned arg1; // index in tape
} Tensor;

typedef struct WengertList {
	Tensor *tensors;
	unsigned size;
	unsigned used;
} WengertList;

#define WengerList_FROM_ARRAY(a) \
	((WengertList){.tensors = (a), .size = ARRAY_LEN(a)})

static bool
Tensor_invariant(const Tensor *t) {
	if (!t) return false;
	// NOTE: if before operating on the tensors I check the number of elements I can relax this.
	if (!IFF(t->rows == 0, t->cols == 0)) return false;
	if (!IFF(t->value == NULL, t->grad == NULL)) return false;

	// Overflow detection.
	unsigned numel = t->rows * t->cols;
	if (t->cols != 0 && numel / t->cols != t->rows) return false;

	// If !Operation_IS_BINARY(t->op) the values of parent0, parent1/scalar are
	// don't-care.
	return true;
}

static bool
WengertList_invariant(const WengertList *tape) {
	if (!tape) return false;
	if (tape->tensors == NULL && tape->size != 0) return false;
	if (tape->used > tape->size) return false;
	// If tape->size == 0 than tape->tensors is don't-care.

	// The first tensor must be an empty one.
	static const unsigned char empty_tensor[sizeof (Tensor)] = {0};
	assert((Tensor *) empty_tensor);
	if (tape->used > 0
		&& memcmp(tape->tensors, empty_tensor, sizeof empty_tensor) != 0)
		return false;

	// The tape must be composed of valid tensors topologically sorted.
	for (unsigned i = 0; i < tape->used; i++) {
		Tensor *t = tape->tensors + i;
		if (!Tensor_invariant(t)) return false;
		if (t->arg0 > i) return false;
		if (Operation_IS_BINARY(t->op) && t->arg1 > i) return false;
		// TODO: determine if it is fine that a parent is an empty tensor
		// (given the fact that nop thensors must have the empty tensor as parents).
	}
	return true;
}

static void
WengertList_append(WengertList tape[static 1], const Tensor t) {
	assert(WengertList_invariant(tape));
	tape->tensors[tape->used++] = t;
	assert(WengertList_invariant(tape));
}

static void
WengertList_reset(WengertList tape[static 1]) {
	assert(WengertList_invariant(tape));
	tape->used = 0;
	assert(WengertList_invariant(tape));
}

static bool
Tensor_same_shape(const Tensor a[static 1], const Tensor b[static 1]) {
	assert(Tensor_invariant(a));
	assert(Tensor_invariant(b));
	return a->rows == b->rows && a->cols == b->cols;
}

static float
sigmoid(float x) {
	return 1.f/(1.f+expf(-x));
}

#include <stdint.h>
#include <limits.h>

// There are four kind of errors right now shape_missmatch, tape_oom, arena_oom
// and propagation (from previous errors).
// typedef struct ExpectedError {unsigned id; Error code;} ExpectedError;
// It would be better to accumulate errors "off-band" with a linked list.

static unsigned
Tensor_new(
	Arena alloc[static 1],
	WengertList tape[static 1],
	float value,
	unsigned rows,
	unsigned cols
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	
	unsigned numel = rows*cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	cblas_scopy(numel, (float[]){value}, 0, data,                1);
	cblas_scopy(numel, (float[]){0},     0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = rows, .cols = cols,
		.op = Operation_nop, .arg0 = 0, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Tensor_sum(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Tensor *x_tensor = tape->tensors + x;

	unsigned numel = 1;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	*data = cblas_sdot(x_tensor->rows*x_tensor->cols,  (float[]){1}, 0, x_tensor->value, 1);
	*(data+numel) = 0;
	// cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = 1, .cols = 1,
		.op = Operation_sum, .arg0 = x, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Tensor_sigmoid(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Tensor *x_tensor = tape->tensors + x;

	unsigned numel = x_tensor->rows*x_tensor->cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = sigmoid(x_tensor->value[i]);
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = x_tensor->rows, .cols = x_tensor->cols,
		.op = Operation_sig, .arg0 = x, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Tensor_negate(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Tensor *x_tensor = tape->tensors + x;

	unsigned numel = x_tensor->rows*x_tensor->cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = -x_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = x_tensor->rows, .cols = x_tensor->cols,
		.op = Operation_neg, .arg0 = x, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

// TODO: broadcast semantic
static unsigned
Tensor_add(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	
	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}
	
	Tensor *lhs_tensor = tape->tensors + lhs;
	Tensor *rhs_tensor = tape->tensors + rhs;
	if (!Tensor_same_shape(lhs_tensor, rhs_tensor)) {
		return 0;
	}
	
	unsigned numel = lhs_tensor->rows*lhs_tensor->cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	// Here using BLAS does not make much sense since we would have to call
	// scopy first and that saxpy.
	for (unsigned i = 0; i < numel; i++) {
		data[i] = lhs_tensor->value[i] + rhs_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);
	
	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = lhs_tensor->rows, .cols = lhs_tensor->cols,
		.op = Operation_add,
		.arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

// TODO: broadcast semantic
static unsigned
Tensor_mul(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Tensor *lhs_tensor = tape->tensors + lhs;
	Tensor *rhs_tensor = tape->tensors + rhs;
	if (!Tensor_same_shape(lhs_tensor, rhs_tensor)) {
		return 0;
	}

	unsigned numel = lhs_tensor->rows*lhs_tensor->cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	// Here using BLAS does not make much sense since we would have to call
	// scopy first and that sbmv.
	for (unsigned i = 0; i < numel; i++) {
		data[i] = lhs_tensor->value[i] * rhs_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = lhs_tensor->rows, .cols = lhs_tensor->cols,
		.op = Operation_mul, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Tensor_pow(
	Arena alloc[static 1],
	WengertList tape[static 1],
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Tensor *lhs_tensor = tape->tensors + lhs;
	Tensor *rhs_tensor = tape->tensors + rhs;
	if (rhs_tensor->rows*rhs_tensor->cols != 1) {
		return 0;
	}

	unsigned numel = lhs_tensor->rows*lhs_tensor->cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = powf(lhs_tensor->value[i], rhs_tensor->value[0]);
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = lhs_tensor->rows, .cols = lhs_tensor->cols,
		.op = Operation_pow, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Tensor_matmul(Arena alloc[static 1],
			  WengertList tape[static 1],
			  unsigned lhs,
			  unsigned rhs) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	
	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Tensor *lhs_tensor = tape->tensors + lhs;
	Tensor *rhs_tensor = tape->tensors + rhs;
	if (lhs_tensor->cols != rhs_tensor->rows) {
		return 0;
	}

	unsigned lhs_rows = lhs_tensor->rows, rhs_cols = rhs_tensor->cols,
		lhs_cols = lhs_tensor->cols;
	// MxK * KxN = MxN
	unsigned M = lhs_rows, N = rhs_cols, K = lhs_cols;
	unsigned numel = M*N;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	float alpha = 1, beta = 0;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
				alpha, lhs_tensor->value, lhs_cols, rhs_tensor->value, rhs_cols,
				beta, data, rhs_cols);
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Tensor t = {
		.value = data, .grad = data + numel,
		.rows = lhs_rows, .cols = rhs_cols,
		.op = Operation_dot, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

// implementare backprop, fare la parte del broadcasting, scrivere dei test con
// vari tensori vuoti, dimensioni broadcasting e ovviemente precisione numerica.
// Implementare il linear layer!
// TIME TO SLAY THE DRAGON!

static void
Tensor_backprop(WengertList tape[static 1]) {
	assert(WengertList_invariant(tape));
	if (tape->used == 0) return;
	Tensor *starting_point = tape->tensors + (tape->used-1);
	assert(starting_point->rows == 1 && starting_point->cols == 1);

	/* Here we accumulate (saxpy) the gradient (vector-Jacobian product) in the
	 * parents of the various operators. The important thing is accumulating the
	 * result of the vector-Jacobian product (we mean vector in the vector space
	 * sense so a matrix is a vector), this is because this operation can (in
	 * our case) always be done without instanciating big (sparse) tensors.
	 *
	 * let z = g(y_1, ..., y_k)
	 * for i = 1,...,k:
	 *     jvp = grad[z] * ∂g/∂y_i
	 *     grad[y_i] += jvp
	 *
	 */
	*starting_point->grad = 1;
	for (unsigned i = tape->used; i-- > 0;) {
		Tensor *t = tape->tensors + i;
		Tensor *arg0 = tape->tensors + t->arg0;
		Tensor *arg1 = tape->tensors + t->arg1;
		switch (t->op) {
			case Operation_nop:
				break;
			case Operation_sum: {
				// z = sum(y)
				// jvp = grad[z] * 1
				// grad[y] += jvp
				float alpha = 1;
				int incx = 0, incy = 1;
				unsigned numel = arg0->rows*arg0->cols;
				assert(t->rows*t->cols == 1);
				cblas_saxpy(numel, alpha, t->grad, incx, arg0->grad, incy);
			} break;
			case Operation_sig: {
				// z = 𝜎(y)
				// jvp = grad[z] * z⋅(1-z)
				// grad[y] += jvp;
				unsigned numel = t->rows*t->cols;
				for (unsigned i = 0; i < numel; i++) {
					arg0->grad[i] += t->grad[i] * t->value[i]*(1-t->value[i]);
				}
			} break;
			case Operation_neg: {
				// z = -y
				// jvp = grad[z] * (-1)
				// grad[y] += jvp
				float alpha = -1;
				int incx = 1, incy = 1;
				unsigned numel = t->rows*t->cols;
				cblas_saxpy(numel, alpha, t->grad, incx, arg0->grad, incy);
			} break;
			// TODO: support broadcasting in backpropagation.
			case Operation_add: {
				// z = y_1 + y_2
				// jvp_1 = grad[z] * (1 + 0)
				// jvp_2 = grad[z] * (0 + 1)
				// grad[y_1] += jvp_1
				// grad[y_2] += jvp_2
				float alpha = 1;
				int incx = 1, incy = 1;
				unsigned numel = t->rows*t->cols;

				cblas_saxpy(numel, alpha, t->grad, incx, arg0->grad, incy);
				cblas_saxpy(numel, alpha, t->grad, incx, arg1->grad, incy);
			} break;
			case Operation_mul: {
				// z = y_1*y_2
				// jvp_1 = grad[z] * 1*y_2
				// jvp_2 = grad[z] * y_1*1
				// grad[y_1] += jvp_1
				// grad[y_2] += jvp_2
				unsigned numel = t->rows*t->cols;

				// cblas_sgbmv https://stackoverflow.com/a/13433038
				for (unsigned i = 0; i < numel; i++) {
					arg0->grad[i] += t->grad[i] * arg1->value[i];
				}
				for (unsigned i = 0; i < numel; i++) {
					arg1->grad[i] += t->grad[i] * arg0->value[i];
				}
			} break;
			case Operation_pow: {
				// z = y_1^y_2
				// jvp_1 = grad[z] * y_2*y_1^(y_2-1)
				// jvp_2 = grad[z] * z * ln(y_1)
				// grad[y_1] += jvp_1
				// grad[y_2] += jvp_2

				unsigned numel = t->rows*t->cols;
				for (unsigned i = 0; i < numel; i++) {
					arg0->grad[i] += t->grad[i] * arg1->value[0] * powf(arg0->value[i], arg1->value[0]-1);
				}
				// https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py#L37
				// https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py#L31
				// Basically due to the total derivative when we broadcast we
				// sum the contributions of all the broadcasted/copied values.
				// Here the broadcasting is implicit in the sense that the
				// single value of the exponenet is broadcasted in a matrix of
				// the same shape of arg0 so that every number in that matrix
				// has its own exponent.
				//                ⎡e e⎤
				//      e         ⎣e e⎦
				// ⎡a b⎤  =  ⎡a b⎤
				// ⎣c d⎦  =  ⎣c d⎦
				// Is like when we do this in scalar code
				// e = something
				// x1 = a^e
				// x2 = b^e
				// x3 = c^e
				// x4 = d^e
				// When back propagating through this we have to sum all the
				// contibutions of the various expontials since we reuse e.
				for (unsigned i = 0; i < numel; i++) {
					arg1->grad[0] += t->grad[i] * t->value[0] * logf(arg0->value[i]);
				}
			} break;
			case Operation_dot: {
				// https://pdfs.semanticscholar.org/c74c/5e11ed05246c12165ce7e4b6222bd32d68dc.pdf
				// z = y_1 y_2
				// jvp_1 = grad[z] * y_2^T
				// jpv_2 = y_1^T * grad[z]
				// y_1 += jvp_1
				// y_2 += jvp_2

				float alpha = 1, beta = 1; // β=1 to perform the accumulation.
				// MxK * KxN = MxN
				unsigned M = t->rows, N = arg1->rows, K = t->cols;
				unsigned LDA = K, LDB = K, LDC = N;
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
							t->grad, LDA, arg1->value, LDB, beta, arg0->grad, LDC);
				unsigned M_ = arg0->cols, N_ = t->cols, K_ = arg0->rows;
				unsigned LDA_ = M_, LDB_ = N_, LDC_ = N_;
				cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M_, N_, K_, alpha,
							arg0->value, LDA_, t->grad, LDB_, beta, arg1->grad, LDC_);
			} break;
		}
	}
}

void test_linear_layer_and_mse_loss(
	float *Wgrad, float *Xgrad, float *Bgrad, float *Ygrad, float *TWOgrad
) {
	Arena *alloc = &(Arena){
		.data = calloc(4 * (1 << 10), 1),
		.size = 4 * (1 << 10),
		.used = 0,
	};
	if (!alloc->data) {
		perror("calloc");
		return 1;
	}

	WengertList *tape = &WengerList_FROM_ARRAY((Tensor [1 << 7]){0});
	tape->used++; // To create the empty tensor.

	// sum((sigmoid(W@X + B) - Y)^2)
	unsigned W = Tensor_new(alloc, tape, 1, 3, 3);
	unsigned X = Tensor_new(alloc, tape, 2, 3, 2);
	unsigned B = Tensor_new(alloc, tape, 3, 3, 2);
	unsigned Y = Tensor_new(alloc, tape, 4, 3, 2);
	unsigned TWO = Tensor_new(alloc, tape, 2, 1, 1);

	unsigned linear = Tensor_sigmoid(alloc, tape,
		Tensor_add(alloc, tape, Tensor_matmul(alloc, tape, W, X), B));

	unsigned L = Tensor_sum(alloc, tape, Tensor_pow(alloc, tape,
		Tensor_add(alloc, tape, linear, Tensor_negate(alloc, tape, Y)), TWO));

	assert(L != 0);
	Tensor_backprop(tape);

	int INCY = 1, INCX = 1;
	cblas_scopy(3*3, tape->tensors[W].grad, INCX, Wgrad, INCY);
	cblas_scopy(3*2, tape->tensors[X].grad, INCX, Xgrad, INCY);
	cblas_scopy(3*2, tape->tensors[B].grad, INCX, Bgrad, INCY);
	cblas_scopy(3*2, tape->tensors[Y].grad, INCX, Ygrad, INCY);
	cblas_scopy(1*1, tape->tensors[TWO].grad, INCX, TWOgrad, INCY);
}

#ifdef TEST

#include <stdlib.h>
#include <stdio.h>

int main(void) {
	Arena *alloc = &(Arena){
		.data = calloc(4 * (1 << 10), 1),
		.size = 4 * (1 << 10),
		.used = 0,
	};
	if (!alloc->data) {
		perror("calloc");
		return 1;
	}
	
	WengertList *tape = &WengerList_FROM_ARRAY((Tensor [1 << 7]){0});
	tape->used++; // To create the empty tensor.

	/* A = torch.ones(3, 2, requires_grad=True)
	 * x = torch.ones(2, 1)*2
	 * b = torch.ones(3, 1)*2
	 * y = A@x + b
	 * L = sum(torch.nn.functional.sigmoid(y))
	 * L.backward()
	 * A.grad
	 * tensor([[0.0049, 0.0049],
	 *         [0.0049, 0.0049],
	 *         [0.0049, 0.0049]])
	 */

	unsigned A = Tensor_new(alloc, tape, 1, 3, 2);
	unsigned x = Tensor_new(alloc, tape, 2, 2, 1);
	unsigned b = Tensor_new(alloc, tape, 2, 3, 1);

	// y = Ax + b
	unsigned y = Tensor_add(alloc, tape, Tensor_matmul(alloc, tape, A, x), b);
	unsigned L = Tensor_sum(alloc, tape, Tensor_sigmoid(alloc, tape, y));

	assert(L != 0);
	Tensor_backprop(tape);

	cblas_sprint(CblasRowMajor, CblasNoTrans, 3, 2, tape->tensors[A].grad, tape->tensors[A].cols);

	// TODO: test back propagation trough negation, Hadamard product and power.
	// TODO: how do I tell my ligrary to backpropagate only in the weights?
	// The resilt that I get in x.grad are non sensical or is it fine???
	return 0;
}

#endif
