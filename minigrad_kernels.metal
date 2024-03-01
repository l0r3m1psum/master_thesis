#include <metal_atomic>
#include <metal_math>

#define COARSE_FACTOR 4

uint3 threadIdx [[thread_position_in_threadgroup]];
uint3 blockDim  [[threads_per_threadgroup]];
uint3 blockIdx  [[threadgroup_position_in_grid]];
uint3 gridDim   [[threads_per_grid]];

// TODO: make this simpler...
// also support vectors with non even length
// https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu#L179
[[kernel]] void
sumFwd(
	const device float                *input    [[buffer(0)]],
	device       metal::atomic<float> *output   [[buffer(1)]],
	threadgroup  float                *input_tg [[threadgroup(0)]]
) {
	metal::mem_flags mem_flag = metal::mem_flags::mem_none;
	constexpr metal::memory_order mem_order = metal::memory_order::memory_order_relaxed;

	uint segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
	uint tx = threadIdx.x, i = segment + tx;

	atomic_store_explicit(output, 0.f, mem_order); // just in case there was trash there.
	float sum = input[i];
	for (uint tile = 1; tile < COARSE_FACTOR*2; tile++) {
		sum += input[i + tile*blockDim.x];
	}

	input_tg[tx] = sum;
	for (uint stride = blockDim.x/2; stride >= 1; stride /= 2) {
		threadgroup_barrier(mem_flag);
		if (tx < stride) {
			input_tg[tx] += input_tg[tx + stride];
		}
	}
	if (tx == 0) {
		(void) atomic_fetch_add_explicit(output, input_tg[0], mem_order);
	}
}
[[kernel]] void
sigFwd(
   const device float *input  [[buffer(0)]],
		 device float *output [[buffer(1)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	output[i] = 1.f/(1.f+metal::exp(-input[i]));
}
[[kernel]] void
negFwd(
   const device float *input  [[buffer(0)]],
		 device float *output [[buffer(1)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	output[i] = -input[i];
}
[[kernel]] void
addFwd(
   const device float *lhs    [[buffer(0)]],
   const device float *rhs    [[buffer(1)]],
		 device float *output [[buffer(2)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	output[i] = lhs[i] + rhs[i];
}
[[kernel]] void
mulFwd(
	const device float *lhs    [[buffer(0)]],
	const device float *rhs    [[buffer(1)]],
		  device float *output [[buffer(2)]]
 ) {
	 uint i = blockDim.x*blockIdx.x + threadIdx.x;
	 output[i] = lhs[i] * rhs[i];
 }
[[kernel]] void
powFwd(
	const device float *lhs    [[buffer(0)]],
	const device float *rhs    [[buffer(1)]],
		  device float *output [[buffer(2)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	output[i] = metal::pow(lhs[i], *rhs);
}
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-001.html
// C := op(A)*op(B)
// where op(X) is either op(X) = X or op(X) = X^T
// op(A) is an m-by-k matrix, (dimentions are after op is applied!)
// op(B) is a k-by-n matrix,
// C is an m-by-n matrix.
[[kernel]] void
dotFwd( // we implicitly do an MxN loop
	constant     bool  &transa [[buffer(0)]],
	constant     bool  &transb [[buffer(1)]],
	constant     uint  &K      [[buffer(2)]],
	device const float *A      [[buffer(3)]], // MxK
	device const float *B      [[buffer(4)]], // KxN
	device       float *C      [[buffer(5)]]  // MxN
) {
	uint M = gridDim.x; uint N = gridDim.y;
	uint row = blockDim.x*blockIdx.x + threadIdx.x;
	uint col = blockDim.y*blockIdx.y + threadIdx.y;

	// An MxN (ROWSxCOLS) matrix will always occupy M*N slots of memory
	// invariant of transposition. Now let A be a 3x2 matrix, we can visualize
	// it and its transpose, with as elements the offset in memory from A's base
	// address, as such:
	//     |0|1|        |0|2|4|
	// A = |2|3|  A^T = |1|3|5|
	//     |4|5|
	// Now A(x, y) = x*N + y because to change row we need to go forward by N=2
	// elements, while A^T(x, y) = x + y*N because to change column we need to
	// go forward by N=2 elements.

	float res = 0;
	if (!transa && !transb) {
		for (uint k = 0; k < K; k++) {
			res += A[row*K + k] * B[k*K + col];
		}
	} else if (transa && !transb) {
		for (uint k = 0; k < K; k++) {
			res += A[row + k*K] * B[k*K + col];
		}
	} else if (!transa && transb) {
		for (uint k = 0; k < K; k++) {
			res += A[row*K + k] * B[k + col*K];
		}
	} else /* (transa && transb) */ {
		for (uint k = 0; k < K; k++) {
			res += A[row + k*K] * B[k + col*K];
		}
	}
	C[row*M + col] = res;
}

[[kernel]] void sumBwd() {} // saxpy
[[kernel]] void
sigBwd(
	      device float *a [[buffer(0)]],
	const device float *b [[buffer(1)]],
	const device float *c [[buffer(2)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	a[i] += b[i] * c[i]*(1.f-c[i]);
}

[[kernel]] void
mulBwd() {}

[[kernel]] void
powArg0Bwd(
	      device float  *a [[buffer(0)]],
	const device float  *b [[buffer(1)]],
		constant float&  c [[buffer(2)]],
	const device float  *d [[buffer(3)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	a[i] += b[i] * c * metal::pow(d[i], c-1);
}

[[kernel]] void
powArg1Bwd(
	      device float&  a [[buffer(0)]],
	const device float  *b [[buffer(1)]],
	const device float  *c [[buffer(2)]],
	const device float  *d [[buffer(3)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	a += b[i] * c[i] * metal::log(d[i]);
}

// saxpy
[[kernel]] void
update(
	    constant float&  alpha [[buffer(0)]],
	const device float  *x     [[buffer(1)]],
	constant     uint&   incx  [[buffer(2)]],
	      device float  *y     [[buffer(3)]],
	    constant uint&   incy  [[buffer(4)]]
) {
	uint i = blockDim.x*blockIdx.x + threadIdx.x;
	y[i*incy] += alpha*x[i*incx];
}
