#define IFF(a, b) (!!(a) == !!(b))
#define ARRAY_LEN(a) (sizeof (a) / sizeof *(a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#pragma mark BLAS (Basic Linear Algebra Subroutine)

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include "blas.h"
#undef ACCELERATE_NEW_LAPACK
#undef ACCELERATE_LAPACK_ILP64

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
	Operation_neg, // negation
	// 2-arity
	Operation_add, // element-wise (with broadcasting)
	Operation_mul, // element-wise (with broadcasting)
	Operation_pow, // element-wise (only by scalar)
	Operation_dot, // matrix multiplication (without vector promotion)
} Operation;

#define Operation_IS_BINARY(op) ((op) >= Operation_add)

// NOTE: probably passing tensor by value is even better due to the way in which the calling convention works.
typedef struct Mat {
	float *value, *grad; // always the same shape.
	unsigned rows, cols;
	// recipe
	Operation op;
	unsigned arg0; // index in tape
	unsigned arg1; // index in tape
} Mat;

typedef struct WengertList {
	Mat *mats;
	unsigned size;
	unsigned used;
} WengertList;

#define WengerList_FROM_ARRAY(a) \
	((WengertList){.mats = (a), .size = ARRAY_LEN(a)})

static bool
Mat_invariant(const Mat *t) {
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
	if (tape->mats == NULL && tape->size != 0) return false;
	if (tape->used > tape->size) return false;
	// If tape->size == 0 than tape->tensors is don't-care.

	// The first tensor must be an empty one.
	static const unsigned char empty_tensor[sizeof (Mat)] = {0};
	assert((Mat *) empty_tensor);
	if (tape->used > 0
		&& memcmp(tape->mats, empty_tensor, sizeof empty_tensor) != 0)
		return false;

	// The tape must be composed of valid tensors topologically sorted.
	for (unsigned i = 0; i < tape->used; i++) {
		Mat *t = tape->mats + i;
		if (!Mat_invariant(t)) return false;
		if (t->arg0 > i) return false;
		if (Operation_IS_BINARY(t->op) && t->arg1 > i) return false;
		// TODO: determine if it is fine that a parent is an empty tensor
		// (given the fact that nop thensors must have the empty tensor as parents).
	}
	return true;
}

static void
WengertList_append(WengertList tape[static 1], const Mat t) {
	assert(WengertList_invariant(tape));
	tape->mats[tape->used++] = t;
	assert(WengertList_invariant(tape));
}

static void
WengertList_reset(WengertList tape[static 1]) {
	assert(WengertList_invariant(tape));
	tape->used = 0;
	assert(WengertList_invariant(tape));
}

static bool
Mat_same_shape(const Mat a[static 1], const Mat b[static 1]) {
	assert(Mat_invariant(a));
	assert(Mat_invariant(b));
	return a->rows == b->rows && a->cols == b->cols;
}

static unsigned
Mat_numel(const Mat a[static 1]) {
	assert(Mat_invariant(a));
	return a->rows*a->cols;
}

static float
sigmoid(float x) {
	return 1.f/(1.f+expf(-x));
}

// This is the global context.
Arena *alloc;
WengertList *tape;

#include <stdint.h>
#include <limits.h>

// There are four kind of errors right now shape_missmatch, tape_oom, arena_oom
// and propagation (from previous errors).
// typedef struct ExpectedError {unsigned id; Error code;} ExpectedError;
// It would be better to accumulate errors "off-band" with a linked list.

typedef struct Size {
	unsigned rows, cols;
} MSize;

static unsigned
Mat_new(
	float value,
	MSize size
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	unsigned rows = size.rows, cols = size.cols;

	unsigned numel = rows*cols;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	cblas_scopy(numel, (float[]){value}, 0, data,                1);
	cblas_scopy(numel, (float[]){0},     0, data+numel, 1);

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = rows, .cols = cols,
		.op = Operation_nop, .arg0 = 0, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Mat_sum(
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Mat *x_tensor = tape->mats + x;

	unsigned numel = 1;
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	*data = cblas_sdot(Mat_numel(x_tensor), (float[]){1}, 0, x_tensor->value, 1);
	*(data+numel) = 0;
	// cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = 1, .cols = 1,
		.op = Operation_sum, .arg0 = x, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Mat_sigmoid(
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Mat *x_tensor = tape->mats + x;

	unsigned numel = Mat_numel(x_tensor);
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = sigmoid(x_tensor->value[i]);
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = x_tensor->rows, .cols = x_tensor->cols,
		.op = Operation_sig, .arg0 = x, .arg1 = 0,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Mat_negate(
	unsigned x
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (x >= tape->used) {
		return 0;
	}

	Mat *x_tensor = tape->mats + x;

	unsigned numel = Mat_numel(x_tensor);
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = -x_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Mat t = {
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
Mat_add(
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	
	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}
	
	Mat *lhs_tensor = tape->mats + lhs;
	Mat *rhs_tensor = tape->mats + rhs;
	if (!Mat_same_shape(lhs_tensor, rhs_tensor)) {
		return 0;
	}
	
	unsigned numel = Mat_numel(lhs_tensor);
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	// Here using BLAS does not make much sense since we would have to call
	// scopy first and that saxpy.
	for (unsigned i = 0; i < numel; i++) {
		data[i] = lhs_tensor->value[i] + rhs_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);
	
	Mat t = {
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
Mat_mul(
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Mat *lhs_tensor = tape->mats + lhs;
	Mat *rhs_tensor = tape->mats + rhs;
	if (!Mat_same_shape(lhs_tensor, rhs_tensor)) {
		return 0;
	}

	unsigned numel = Mat_numel(lhs_tensor);
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	// Here using BLAS does not make much sense since we would have to call
	// scopy first and that sbmv.
	for (unsigned i = 0; i < numel; i++) {
		data[i] = lhs_tensor->value[i] * rhs_tensor->value[i];
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = lhs_tensor->rows, .cols = lhs_tensor->cols,
		.op = Operation_mul, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Mat_pow(
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));

	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Mat *lhs_tensor = tape->mats + lhs;
	Mat *rhs_tensor = tape->mats + rhs;
	if (rhs_tensor->rows*rhs_tensor->cols != 1) {
		return 0;
	}

	unsigned numel = Mat_numel(lhs_tensor);
	float *data = Arena_alloc(alloc, sizeof *data * numel*2);
	if (!data) return 0;

	for (unsigned i = 0; i < numel; i++) {
		data[i] = powf(lhs_tensor->value[i], rhs_tensor->value[0]);
	}
	cblas_scopy(numel, (float[]){0}, 0, data+numel, 1);

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = lhs_tensor->rows, .cols = lhs_tensor->cols,
		.op = Operation_pow, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static unsigned
Mat_matmul(
	unsigned lhs,
	unsigned rhs
) {
	assert(Arena_invariant(alloc));
	assert(WengertList_invariant(tape));
	
	if (lhs >= tape->used || rhs >= tape->used) {
		return 0;
	}

	Mat *lhs_tensor = tape->mats + lhs;
	Mat *rhs_tensor = tape->mats + rhs;
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

	Mat t = {
		.value = data, .grad = data + numel,
		.rows = lhs_rows, .cols = rhs_cols,
		.op = Operation_dot, .arg0 = lhs, .arg1 = rhs,
	};
	WengertList_append(tape, t);
	assert(WengertList_invariant(tape));
	return tape->used - 1;
}

static bool
Mat_backprop(unsigned l) {
	assert(WengertList_invariant(tape));

	if (l >= tape->used) {
		return false;
	}
	Mat *starting_point = tape->mats + (tape->used-1);
	if (Mat_numel(starting_point) != 1) {
		return false;
	}

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
	for (unsigned i = l+1; i-- > 0;) {
		Mat *t = tape->mats + i;
		Mat *arg0 = tape->mats + t->arg0;
		Mat *arg1 = tape->mats + t->arg1;
		switch (t->op) {
			case Operation_nop:
				break;
			case Operation_sum: {
				// z = sum(y)
				// jvp = grad[z] * 1
				// grad[y] += jvp
				float alpha = 1;
				int incx = 0, incy = 1;
				assert(Mat_numel(t) == 1);
				cblas_saxpy(Mat_numel(arg0), alpha, t->grad, incx, arg0->grad, incy);
			} break;
			case Operation_sig: {
				// z = 𝜎(y)
				// jvp = grad[z] * z⋅(1-z)
				// grad[y] += jvp;
				unsigned numel = Mat_numel(t);
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
				unsigned numel = Mat_numel(t);
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
				unsigned numel = Mat_numel(t);

				cblas_saxpy(numel, alpha, t->grad, incx, arg0->grad, incy);
				cblas_saxpy(numel, alpha, t->grad, incx, arg1->grad, incy);
			} break;
			case Operation_mul: {
				// z = y_1*y_2
				// jvp_1 = grad[z] * 1*y_2
				// jvp_2 = grad[z] * y_1*1
				// grad[y_1] += jvp_1
				// grad[y_2] += jvp_2
				unsigned numel = Mat_numel(t);

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

				unsigned numel = Mat_numel(t);
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

	return true;
}

void test_linear_layer_and_mse_loss(
	float *Wgrad, float *Xgrad, float *Bgrad, float *Ygrad, float *TWOgrad
) {
	// FIXME: this is stack allocated too...
	alloc = &(Arena){
		.data = calloc(4 * (1 << 10), 1),
		.size = 4 * (1 << 10),
		.used = 0,
	};
	if (!alloc->data) {
		perror("calloc");
		return 1;
	}

	// FIXME: this makes the global tape point to stack memory that will not be
	// valid after this function terminate!
	tape = &WengerList_FROM_ARRAY((Mat [1 << 7]){0});
	tape->used++; // To create the empty tensor.

	// sum((sigmoid(W@X + B) - Y)^2)
	unsigned W = Mat_new(1, (MSize){3, 3});
	unsigned X = Mat_new(2, (MSize){3, 2});
	unsigned B = Mat_new(3, (MSize){3, 2});
	unsigned Y = Mat_new(4, (MSize){3, 2});
	unsigned TWO = Mat_new(2, (MSize){1, 1});

	unsigned linear = Mat_sigmoid(Mat_add(Mat_matmul(W, X), B));

	unsigned L = Mat_sum(Mat_pow(Mat_add(linear, Mat_negate(Y)), TWO));

	assert(L != 0);
	Mat_backprop(L);

	int INCY = 1, INCX = 1;
	cblas_scopy(3*3, tape->mats[W].grad, INCX, Wgrad, INCY);
	cblas_scopy(3*2, tape->mats[X].grad, INCX, Xgrad, INCY);
	cblas_scopy(3*2, tape->mats[B].grad, INCX, Bgrad, INCY);
	cblas_scopy(3*2, tape->mats[Y].grad, INCX, Ygrad, INCY);
	cblas_scopy(1*1, tape->mats[TWO].grad, INCX, TWOgrad, INCY);
}

// TODO: support req_grad=False, also we need to propagrate this property iff all args to an operator are req_grad=False

#ifdef TEST_minigrad

#include <stdlib.h>
#include <stdio.h>

int main(void) {
	// FIXME: stack allocated.
	alloc = &(Arena){
		.data = calloc(4 * (1 << 10), 1),
		.size = 4 * (1 << 10),
		.used = 0,
	};
	if (!alloc->data) {
		perror("calloc");
		return 1;
	}
	
	// FIXME: stack allocated.
	tape = &WengerList_FROM_ARRAY((Mat [1 << 7]){0});
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

	unsigned A = Mat_new(1, (MSize){3, 2});
	unsigned x = Mat_new(2, (MSize){2, 1});
	unsigned b = Mat_new(2, (MSize){3, 1});

	// y = Ax + b
	unsigned y = Mat_add(Mat_matmul(A, x), b);
	unsigned L = Mat_sum(Mat_sigmoid(y));

	assert(L != 0);
	Mat_backprop(L);

	cblas_sprint(CblasRowMajor, CblasNoTrans, 3, 2, tape->mats[A].grad, tape->mats[A].cols);

	// TODO: test back propagation trough Hadamard product.
	return 0;
}

#endif

#if 0
Mat Mat_op(Mat lhs, Mat rhs) {
	if (!Mat_same_shape(lhs, rhs)) {
		return 0;
	}

	Mat res = Mat_empty(same, shape); // data, grad, rows, cols, op, arg0, arg1
	// In mat_empty we should also record the reason for the error:
	//   tape_oom, arena_oom
	// Two errors are now implicit shape_mismatch and propagation. Propagation
	// is fine if it is implicit... But shape missmatch is not.
	for (uint i = 0; i < Mat_numel(res)) {
		...
	}

	return res;
}

void f(void) {
	// This is needed for the global context used for recording the execution
	// and allocating memory.
	Mat_init();

	// We specify the shape and if we require the gradient to be calculated.
	Mat a = Mat_new(3, 3, true), b = Mat_new(3, 1, true);
	// initializing a and b...
	Mat c = Mat_dot(a, b);
	Mat l = Mat_sum(c);
	bool arg_is_scalar = Mat_backward(l);
	assert(arg_is_scalar);
	Mat_grad(a); // returns a float pointer???

	Mat_zero_grad(a); Mat_zero_grad(b);

	// sets no_grad at the context level so that the next operations are not
	// considered for back propagation.
	Mat_no_grad();
	// Remember that we have to build the graph once and then execute it
	// multiple times! Therefore putting everything in a Wengert list is okay
	// since we assume that every calculation is needed.
}
#endif
