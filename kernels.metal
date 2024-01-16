/* Kernel Function Input Attributes
 *
 * +--------------------------------+-----------+
 * | Metal                          | CUDA      |
 * +--------------------------------+-----------+
 * | thread_position_in_threadgroup | threadIdx |
 * | threads_per_threadgroup        | blockDim  |
 * | threadgroup_position_in_grid   | blockIdx  |
 * | threads_per_grid               | gridDim   |
 * | threads_per_simdgroup          | warpSize  |
 * +--------------------------------+-----------+
 *
 * Some Metal meculiarities:
 *   - thread_index_in_threadgroup is the flattened thread_position_in_threadgroup
 *   - thread_position_in_grid
 *       == threads_per_threadgroup * threadgroup_position_in_grid + thread_position_in_threadgroup
 *       == blockDim * blockIdx + threadIdx
 *
 * Address Spaces
 *
 * +-------------+--------------+
 * | Metal       | CUDA         |
 * +-------------+--------------+
 * | device      | __device__   |
 * | threadgroup | __shared__   |
 * | constant    | __constant__ |
 * +-------------+--------------+
 *
 * Synchronization and SIMD-group Functions
 *
 * +--------------------------------------------------------+-----------------+
 * | Metal                                                  | CUDA            |
 * +--------------------------------------------------------+-----------------+
 * | threadgroup_barrier(metal::mem_flags::???)             | __syncthreads() |
 * +--------------------------------------------------------+-----------------+
 * | simdgroup_barrier(metal::mem_flags::???)               | __syncwarp()    |
 * +--------------------------------------------------------+-----------------+
 *
 * https://github.com/xmartlabs/gpgpu-comparison
 * https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups?language=objc
 * https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
 *
 *******************************************************************************
 *
 * GPU Architecture
 *
 * +------------------------+-------------------------------+
 * | AMD/HSA                | Nvidia                        |
 * +------------------------+-------------------------------+
 * | Compute Unit (CU)      | Streaming Multiprocessor (SM) |
 * | Local Data Share (LDS) | Shared Memory                 |
 * | Workgroup              | Thread Block                  |
 * | Grid                   | Grid                          |
 * | Work-item (WI)         | Thread                        |
 * | Wavefront (WF)         | Warp                          |
 * +------------------------+-------------------------------+
 *
 * http://old.gem5.org/wiki/images/1/19/AMD_gem5_APU_simulator_isca_2018_gem5_wiki.pdf
 */

#include <metal_compute>
#include "metal_common.h"

// contains usefull links to CUTLASS documetation https://towardsdatascience.com/matrix-multiplication-on-the-gpu-e920e50207a8
[[kernel]] void
matmul_naive(
	constant float *a     [[buffer(0)]],
	constant float *b     [[buffer(1)]],
	device   float *c     [[buffer(2)]],
	constant uint&  width [[buffer(3)]],
			 uint2  tpg   [[thread_position_in_grid]]
) {
	// This two should be switched.
	uint row = tpg.x, col = tpg.y;
	if ((row < width) & (col < width)) {
		float C = 0;
		for (uint k = 0; k < width; k++) {
			C += a[row*width + k] * b[k*width + col];
		}
		c[row*width + col] = C;
	}
}

uint3 threadIdx [[thread_position_in_threadgroup]];
uint3 blockDim [[threads_per_threadgroup]];
uint3 blockIdx [[threadgroup_position_in_grid]];
uint3 gridDim [[threads_per_grid]];

constant uint TILE_WIDTH [[function_constant(0)]];
constant bool debug_enabled [[function_constant(1)]];
constant bool debug_enabled_defined = is_function_constant_defined(debug_enabled);

#define ASSERT_SCAFFOLDING(n) device int *debug \
	[[buffer((n)),function_constant(debug_enabled_defined)]]
#define ASSERT(cond) do { \
	if (debug_enabled_defined && !(cond)) { \
		uint debug_row = blockIdx.y * blockDim.y + threadIdx.y; \
		uint debug_col = blockIdx.x * blockDim.x + threadIdx.x; \
		uint debug_idx = debug_row*gridDim.x + debug_col; \
		debug[debug_idx] = __LINE__; \
		return; /* Here we should abort or something instead... */ \
	} \
} while(false)

// We have 64KiB for each Local Data Share (i.e. shared memory space per block).
[[kernel]] void
matmul_tiled(
	device const float *M     [[buffer(0)]],
	device const float *N     [[buffer(1)]],
	device       float *P     [[buffer(2)]],
	constant     uint&  Width [[buffer(3)]],
	threadgroup  float *Mtg   [[threadgroup(0)]],
	threadgroup  float *Ntg   [[threadgroup(1)]],
	ASSERT_SCAFFOLDING(4)
) {
	metal::mem_flags mem_flag = metal::mem_flags::mem_none;
	uint bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
	uint Row = by * TILE_WIDTH + ty, Col = bx * TILE_WIDTH + tx;

	ASSERT(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

	float Pvalue = 0;
	for (uint ph = 0; ph < CEIL_DIV(Width, TILE_WIDTH); ph++) {
		uint TileCol = ph*TILE_WIDTH+tx, TileRow = ph*TILE_WIDTH+ty;
		Mtg[ty*TILE_WIDTH + tx] = (Row < Width && TileCol < Width)
			? M[Row*Width + TileCol] : 0.f;
		Ntg[ty*TILE_WIDTH + tx] = (Col < Width && TileRow < Width)
			? N[TileRow*Width + Col] : 0.f;
		threadgroup_barrier(mem_flag);

		for (uint k = 0; k < TILE_WIDTH; k++) {
			Pvalue += Mtg[ty*TILE_WIDTH + k] * Ntg[k*TILE_WIDTH + tx];
		}
		threadgroup_barrier(mem_flag);
	}
	if (Row < Width && Col < Width) {
		P[Row*Width + Col] = Pvalue;
	} else {
		ASSERT(Pvalue == 0.f);
	}
}

// NOTE: using int instead of uint is probabbly better.
[[kernel]] void
matmul_tiled_coarsed(
	device const float *M     [[buffer(0)]],
	device const float *N     [[buffer(1)]],
	device       float *P     [[buffer(2)]],
	constant     uint&  Width [[buffer(3)]],
	threadgroup  float *Mtg   [[threadgroup(0)]],
	threadgroup  float *Ntg   [[threadgroup(1)]],
	ASSERT_SCAFFOLDING(4)
) {
	metal::mem_flags mem_flag = metal::mem_flags::mem_none;
	uint bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
	uint Row = by * TILE_WIDTH + ty, ColStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

	ASSERT(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

	float PValue[COARSE_FACTOR] = {0};
	for (uint ph = 0; ph < CEIL_DIV(Width, TILE_WIDTH); ph++) {
		uint TileCol = ph*TILE_WIDTH+tx, TileRow = ph*TILE_WIDTH+ty;
		Mtg[ty*TILE_WIDTH + tx] = (Row < Width && TileCol < Width)
			? M[Row*Width + TileCol] : 0.f;

		for (uint c = 0; c < COARSE_FACTOR; c++) {
			uint Col = ColStart + c*TILE_WIDTH;
			Ntg[ty*TILE_WIDTH + tx] = (Col < Width && TileRow < Width)
				? N[TileRow*Width + Col] : 0.f;
			threadgroup_barrier(mem_flag);

			for (uint k = 0; k < TILE_WIDTH; k++) {
				PValue[c] += Mtg[ty*TILE_WIDTH + k] * Ntg[k*TILE_WIDTH + tx];
			}
			threadgroup_barrier(mem_flag);
		}
	}

	for (uint c = 0; c < COARSE_FACTOR; c++) {
		uint Col = ColStart + c*TILE_WIDTH;
		if (Row < Width && Col < Width) {
			P[Row*Width + Col] = PValue[c];
		} else {
			ASSERT(PValue[c] == 0.f);
		}
	}
}

#include <metal_atomic>

// This is sdot with a scalar 1 and a stride of 0;
[[kernel]] void
sum(
	const device float                *input    [[buffer(0)]],
	device       metal::atomic<float> *output   [[buffer(1)]],
	threadgroup  float                *input_tg [[threadgroup(0)]],
	ASSERT_SCAFFOLDING(2)
) {
	metal::mem_flags mem_flag = metal::mem_flags::mem_none;
	constexpr metal::memory_order mem_order = metal::memory_order::memory_order_relaxed;

	// Every thread start summing 2 numbers (blockDim.x apart) therefore their
	// block dim is effectively double.
	uint segment = 2*blockDim.x*blockIdx.x;
	uint tx = threadIdx.x, i = segment + tx;

	input_tg[tx] = input[i] + input[i + blockDim.x];
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
sum_coarsed(
	const device float                *input    [[buffer(0)]],
	device       metal::atomic<float> *output   [[buffer(1)]],
	threadgroup  float                *input_tg [[threadgroup(0)]],
	ASSERT_SCAFFOLDING(2)
) {
	metal::mem_flags mem_flag = metal::mem_flags::mem_none;
	constexpr metal::memory_order mem_order = metal::memory_order::memory_order_relaxed;

	uint segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
	uint tx = threadIdx.x, i = segment + tx;

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

#pragma mark Random Number Generation

#include <metal_math>

// Taken from chapter 37 of GPU Gems 3
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// Other resources
// https://stackoverflow.com/questions/3125148/using-random-numbers-with-gpus
// https://towardsdatascience.com/how-to-generate-a-vector-of-random-numbers-on-a-gpu-a37230f887a6
// https://faculty.uml.edu/vbarsegov/gpu/rng/rng.html
// https://diglib.eg.org/xmlui/bitstream/handle/10.2312/EGGH.EGGH06.087-094/087-094.pdf?sequence=1&isAllowed=y
// https://indico.wigner.hu/event/993/contributions/2235/attachments/1800/3186/7_IstvanKiss.pdf
// https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html

static unsigned
TausStep(thread unsigned &z, int S1, int S2, int S3, unsigned M) {
	unsigned b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
}

static unsigned
LCGStep(thread unsigned &z) {
	return z = (1664525 * z + 1013904223);
}

static float
getRandomValueTauswortheUniform(
	thread unsigned &z1,
	thread unsigned &z2,
	thread unsigned &z3,
	thread unsigned &z4
	) {
	unsigned taus = TausStep(z1, 13, 19, 12, 4294967294UL)
		^ TausStep(z2, 2, 25, 4, 4294967288UL)
		^ TausStep(z3, 3, 11, 17, 4294967280UL);
	unsigned lcg = LCGStep(z4);

	return 2.3283064365387e-10f * (taus ^ lcg);
}

static void
boxMuller(float u1, float u2, thread float &uo1, thread float &uo2) {
	float z1 = metal::sqrt(-2.0f * metal::log(u1));
	float s1 = metal::sin(2.0f * M_PI_F * u2);
	float s2 = metal::cos(2.0f * M_PI_F * u2);
	uo1 = z1 * s1;
	uo2 = z1 * s2;
}

static float
getRandomValueTausworthe(
	thread unsigned &z1,
	thread unsigned &z2,
	thread unsigned &z3,
	thread unsigned &z4,
	thread float &temporary,
	unsigned phase
	) {
	if (phase & 1)
	{
		// Return second value of pair
		return temporary;
	}
	else
	{
		float t1, t2, t3;
		// Phase is even, generate pair, return first of values, store second
		t1 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
		t2 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
		boxMuller(t1, t2, t3, temporary);
		return t3;
	}

}

// mul24 is not useful anymore!
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
// https://stackoverflow.com/questions/5544355/cuda-umul24-function-useful-or-not
// Coaleshing is still important! The stack overflow post is lying!
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
// https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved

[[kernel]] void
generateRandomNumbers_tausworthe(
	device   unsigned *seeds           [[buffer(0)]],
	device   float    *output          [[buffer(1)]],
	constant int&      outputs_per_run [[buffer(2)]],
	         uint2     threadIdx       [[thread_position_in_threadgroup]],
	         uint2     blockDim        [[threads_per_threadgroup]],
	         uint2     blockIdx        [[thread_position_in_grid]],
	         uint2     gridDim         [[threads_per_grid]]
	) {
	unsigned z1, z2, z3, z4;
	float temporary;

	// Done this way for coalescing reasons.
	unsigned address = blockIdx.x * blockDim.x + threadIdx.x;
	uint total_num_threads = blockDim.x * gridDim.x;
	z1 = seeds[address + 0 * total_num_threads];
	z2 = seeds[address + 1 * total_num_threads];
	z3 = seeds[address + 2 * total_num_threads];
	z4 = seeds[address + 3 * total_num_threads];

	constexpr int outputs_per_loop = 8;
	for (int loop = 0; loop < outputs_per_run; loop++) {
		unsigned intermediate_address = outputs_per_loop * total_num_threads
			* (loop + blockIdx.x) + threadIdx.x;
		// NOTE: I guess that this could be unrolled.
		for (int i = 0; i < outputs_per_loop; i++) {
			// NOTE: why is the address calculated like this??
			output[intermediate_address + i * blockDim.x]
				= getRandomValueTausworthe(z1, z2, z3, z4, temporary, i);
		}
	}

	// So that if we invoke multiple times we generate different numbers.
	seeds[address + 0 * total_num_threads] = z1;
	seeds[address + 1 * total_num_threads] = z2;
	seeds[address + 2 * total_num_threads] = z3;
	seeds[address + 3 * total_num_threads] = z4;
}
