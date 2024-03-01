#import <Metal/Metal.h>

#define COUNTOF(x) (sizeof (x) / sizeof *(x))
// From Hacker's Delight
// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
#define CEIL_DIV(n, d) ( ( (n) + (d) + ((d)>0?-1:+1) ) / (d) )
#define COARSE_FACTOR 4

typedef enum Op {
	// 0-arity
	Op_nop, // matrix
	// 1-arity
	Op_sum, // reduction
	Op_sig, // sigmoid
	Op_neg, // negation
	// 2-arity
	Op_add, // element-wise
	Op_mul, // element-wise
	Op_pow, // element-wise
	Op_dot, // matrix multiplication
	// meta
	Op_num,
} Op;
const char *Op_str[] = {
	"nop", "sum", "sig", "neg", "add", "mul", "pow", "dot",
};
static_assert(Op_num == COUNTOF(Op_str),
			  "enumeration and names don't match");

id<MTLFunction>             fwdFun[Op_num];
id<MTLComputePipelineState> fwdPSO[Op_num];
id<MTLDevice> dev;
// for backprop
id<MTLFunction>             updateFun, sigBwdFun, powArg0BwdFun, powArg1BwdFun;
id<MTLComputePipelineState> updatePSO, sigBwdPSO, powArg0BwdPSO, powArg1BwdPSO;


typedef struct MShape {
	unsigned rows, cols;
} MShape;

const MShape scalar_shape = {1, 1};

typedef uint Mat;

typedef struct MatRec { // Matrix Record
	enum Op op: 31;
	bool reqGrad: 1;
	Mat arg0, arg1;
	struct MShape shape;
	id<MTLBuffer> fwd, bwd;
} MatRec;

#define MAX_OP 16

uint op_count = 1; // index 0 is the empty matrix used for errors
MatRec tape[MAX_OP];

// With this function we can now uniformly deal with all possible shapes, since
// when we receive a shape with a single dimension to 0 we promote that
// dimension to 1.
// (0,0) -> (0,0)
// (3,0) -> (3,1)
// (0,4) -> (1,4)
// (9,2) -> (9,2)
static MShape
MShape_normalize(MShape s) {
	MShape res = {0};
	res.rows = s.rows + ((s.rows==0)&(s.cols!=0));
	res.cols = s.cols + ((s.cols==0)&(s.rows!=0));
	return res;
}
static size_t
Mshape_numel(MShape s) {
	s = MShape_normalize(s);
	return (size_t)s.rows * (size_t)s.cols;
}

static size_t
MShape_size(MShape s) {
	return sizeof (float) * Mshape_numel(s);
}

static bool
MShape_same(MShape s, MShape t) {
	s = MShape_normalize(s);
	t = MShape_normalize(t);
	return s.rows == t.rows
		&& s.cols == t.cols;

}

static size_t
Mat_numel(Mat arg0) {
	return Mshape_numel(tape[arg0].shape);
}

static bool
Mat_is_param(Mat arg0) {
	MatRec *mat = tape+arg0;
	return mat->op == Op_nop && mat->reqGrad;
}

static bool
Mat_same_shape(Mat arg0, Mat arg1) {
	return MShape_same(tape[arg0].shape, tape[arg1].shape);
}

static Mat
record_entry(MShape shape, bool reqGrad, Op op, Mat arg0, Mat arg1) {
	if (op_count >= MAX_OP) {
		return 0;
	}
	if (arg0 >= op_count || arg1 >= op_count) {
		return 0;
	}

	MTLResourceOptions options = op == Op_nop
		? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
	size_t shape_size = MShape_size(shape);
	id<MTLBuffer> fwd = [dev newBufferWithLength:shape_size options:options];
	id<MTLBuffer> bwd = reqGrad ? [dev newBufferWithLength:shape_size options:options] : nil;

	Mat res = op_count;
	tape[op_count++] = (MatRec){
		.op = op, .reqGrad = reqGrad,
		.arg0 = arg0, .arg1 = arg1,
		.shape = shape,
		.fwd = fwd, .bwd = bwd,
	};
	return res;
}

static Mat
Mat_new(MShape shape, bool reqGrad) {
	return record_entry(shape, reqGrad, Op_nop, 0, 0);
}

static Mat
Mat_sum(Mat arg0) {
	bool res_reqGrad = tape[arg0].reqGrad;
	MShape res_shape = scalar_shape;
	return record_entry(res_shape, res_reqGrad, Op_sum, arg0, 0);
}

static Mat
Mat_sig(Mat arg0) {
	bool res_reqGrad = tape[arg0].reqGrad;
	MShape res_shape = tape[arg0].shape;
	return record_entry(res_shape, res_reqGrad, Op_sig, arg0, 0);
}

static Mat
Mat_neg(Mat arg0) {
	bool res_reqGrad = tape[arg0].reqGrad;
	MShape res_shape = tape[arg0].shape;
	return record_entry(res_shape, res_reqGrad, Op_neg, arg0, 0);
}

static Mat
Mat_add(Mat arg0, Mat arg1) {
	if (!Mat_same_shape(arg0, arg1)) {
		return 0;
	}

	MShape res_shape = tape[arg0].shape;
	bool res_reqGrad = tape[arg0].reqGrad || tape[arg1].reqGrad;
	return record_entry(res_shape, res_reqGrad, Op_add, arg0, arg1);
}

static Mat
Mat_mul(Mat arg0, Mat arg1) {
	if (!Mat_same_shape(arg0, arg1)) {
		return 0;
	}

	MShape res_shape = tape[arg0].shape;
	bool res_reqGrad = tape[arg0].reqGrad || tape[arg1].reqGrad;
	return record_entry(res_shape, res_reqGrad, Op_mul, arg0, arg1);
}

static Mat
Mat_pow(Mat arg0, Mat arg1) {
	MShape exponent_shape = tape[arg1].shape;
	if (exponent_shape.rows != 1 || exponent_shape.cols != 1) {
		return 0;
	}

	MShape res_shape = tape[arg0].shape;
	bool res_reqGrad = tape[arg0].reqGrad || tape[arg1].reqGrad;
	return record_entry(res_shape, res_reqGrad, Op_pow, arg0, arg1);
}

static Mat
Mat_dot(Mat arg0, Mat arg1) {
	// FIXME: I have to chec index first and not inside record_entry
	MShape lhs_shape = tape[arg0].shape, rhs_shape = tape[arg1].shape;
	if (lhs_shape.cols != rhs_shape.rows) {
		return 0;
	}

	MShape res_shape = {.rows = lhs_shape.rows, .cols = rhs_shape.cols};
	bool res_reqGrad = tape[arg0].reqGrad || tape[arg1].reqGrad;
	return record_entry(res_shape, res_reqGrad, Op_dot, arg0, arg1);
}

typedef struct GridSpec {
	MTLSize threadsPerGrid, threadsPerThreadgroup;
} GridSpec;

static GridSpec
getGridSpecElementwiseFun(id<MTLComputePipelineState> pso, NSUInteger size) {
	NSUInteger warp_size = pso.threadExecutionWidth;
	NSUInteger maxThreadPerGroup = pso.maxTotalThreadsPerThreadgroup;

	NSUInteger width = maxThreadPerGroup;
	width -= width%warp_size;
	assert(width%warp_size == 0);

	GridSpec res = {
		.threadsPerGrid        = MTLSizeMake(size, 1, 1),
		.threadsPerThreadgroup = MTLSizeMake(width, 1, 1),
	};
	return res;
}

static void
commitWaitAndCheck(id<MTLCommandBuffer> cmdBuf) {
	[cmdBuf commit];
	[cmdBuf waitUntilCompleted];

	if (cmdBuf.status == MTLCommandBufferStatusError) {
		NSLog(@"%@", cmdBuf.error);
		assert(false);
	}
}

static void
encode_matmul(
	id<MTLCommandBuffer> cmdBuf,
	bool transa, bool transb, NSUInteger M, NSUInteger N, NSUInteger K,
	// No alpha, LDA, LDB or LDC.
	id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C
) {
	NSUInteger w = fwdPSO[Op_dot].threadExecutionWidth;
	NSUInteger h = fwdPSO[Op_dot].maxTotalThreadsPerThreadgroup / w;
	MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
	MTLSize threadsPerGrid = MTLSizeMake(M, N, 1);

	id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
	[cmdEnc setComputePipelineState:fwdPSO[Op_dot]];
	[cmdEnc setBytes:&transa length:sizeof transa atIndex:0];
	[cmdEnc setBytes:&transb length:sizeof transb atIndex:1];
	[cmdEnc setBytes:&K length:sizeof K atIndex:2];
	[cmdEnc setBuffer:A offset:0 atIndex:3]; // input
	[cmdEnc setBuffer:B offset:0 atIndex:4];
	[cmdEnc setBuffer:C offset:0 atIndex:5]; // output
	[cmdEnc dispatchThreads:threadsPerGrid
	  threadsPerThreadgroup:threadsPerThreadgroup];
	[cmdEnc endEncoding];
}

static void
enqueue_forward(id<MTLCommandQueue> cmdQueue) {
	// https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes?language=objc
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
	for (uint i = 0; i < op_count; i++) {
		const MatRec *mat = tape+i;
		id<MTLComputePipelineState> pso = fwdPSO[mat->op];
		NSUInteger warp_size = pso.threadExecutionWidth;
		NSUInteger maxThreadPerGroup = pso.maxTotalThreadsPerThreadgroup;
		switch (mat->op) {
		case Op_nop:
			break;
		case Op_sum: {
			assert(Mat_numel(mat->arg0) % 2 == 0);
			GridSpec spec = getGridSpecElementwiseFun(pso, Mat_numel(mat->arg0));
			spec.threadsPerGrid.width /= 2; // TODO: verify what I'm doing here...

			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			[cmdEnc setComputePipelineState:pso];
			[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:0]; // input
			[cmdEnc setBuffer:mat->fwd offset:0 atIndex:1]; // output
			[cmdEnc setThreadgroupMemoryLength:sizeof (float) * spec.threadsPerThreadgroup.width
									   atIndex:0];
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_sig: {
			GridSpec spec = getGridSpecElementwiseFun(pso, Mat_numel(mat->arg0));

			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			[cmdEnc setComputePipelineState:pso];
			[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:0]; // input
			[cmdEnc setBuffer:mat->fwd offset:0 atIndex:1]; // output
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_neg: {
			GridSpec spec = getGridSpecElementwiseFun(pso, Mat_numel(mat->arg0));

			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			[cmdEnc setComputePipelineState:pso];
			[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:0]; // input
			[cmdEnc setBuffer:mat->fwd            offset:0 atIndex:1]; // output
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_add: {
			GridSpec spec = getGridSpecElementwiseFun(pso, Mat_numel(mat->arg0));

			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			[cmdEnc setComputePipelineState:pso];
			[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:0]; // input
			[cmdEnc setBuffer:tape[mat->arg1].fwd offset:0 atIndex:1];
			[cmdEnc setBuffer:mat->fwd            offset:0 atIndex:2]; // output
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_mul: {
			assert(false);
		} break;
		case Op_pow: {
			GridSpec spec = getGridSpecElementwiseFun(pso, Mat_numel(mat->arg0));

			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			[cmdEnc setComputePipelineState:pso];
			[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:0]; // input
			[cmdEnc setBuffer:tape[mat->arg1].fwd offset:0 atIndex:1];
			[cmdEnc setBuffer:mat->fwd offset:0 atIndex:2]; // output
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_dot: {
			uint M = mat->shape.rows;
			uint N = mat->shape.cols;
			uint K = tape[mat->arg0].shape.cols;
			encode_matmul(cmdBuf, false, false, M, N, K,
						  tape[mat->arg0].fwd, tape[mat->arg1].fwd, mat->fwd);
		} break;
		}
	}

	commitWaitAndCheck(cmdBuf);
}

static void
encode_update(
	id<MTLCommandBuffer> cmdBuf, NSUInteger n,
	float alpha, id<MTLBuffer> x, uint incx, id<MTLBuffer> y, uint incy
) {
	GridSpec spec = getGridSpecElementwiseFun(updatePSO, n);
	id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
	[cmdEnc setComputePipelineState:updatePSO];
	[cmdEnc setBytes:&alpha length:sizeof alpha atIndex:0];
	[cmdEnc setBuffer:x offset:0 atIndex:1];
	[cmdEnc setBytes:&incx length:sizeof incx atIndex:2];
	[cmdEnc setBuffer:y offset:0 atIndex:3];
	[cmdEnc setBytes:&incy length:sizeof incy atIndex:4];
	[cmdEnc dispatchThreads:spec.threadsPerGrid
	  threadsPerThreadgroup:spec.threadsPerThreadgroup];
	[cmdEnc endEncoding];
}

// TODO: take loss as an argument.
static void
enqueue_backward(id<MTLCommandQueue> cmdQueue) {
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];

	assert(Mat_numel(op_count-1) == 1);
	*(float *)tape[op_count-1].bwd.contents = 1.f;

	for (uint i = op_count; i-- > 0;) {
		const MatRec *mat = tape+i;
		if (!mat->reqGrad) continue;
		// For functions with arity 2 we have to check if one of its two
		// arguments does not requires a gradient (and skip propagation in that
		// case). For the arity 1 case the parent (arg0) must require gradient.
		switch (mat->op) {
		case Op_nop:
			break;
		case Op_sum: {
			// i.arg0.grad += i.grad
			// cblas_saxpy(Mat_numel(arg0), 1.f, t->grad, 0, arg0->grad, 1);
			encode_update(cmdBuf, Mat_numel(mat->arg0), 1, mat->bwd, 0, tape[mat->arg0].bwd, 1);
		} break;
		case Op_sig: {
			GridSpec spec = getGridSpecElementwiseFun(sigBwdPSO, Mat_numel(mat->arg0));
			id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
			// arg0->grad[i] += t->grad[i]*t->value[i]*(1.f - t->value[i]);
			[cmdEnc setComputePipelineState:sigBwdPSO];
			[cmdEnc setBuffer:tape[mat->arg0].bwd offset:0 atIndex:0];
			[cmdEnc setBuffer:mat->bwd            offset:0 atIndex:1];
			[cmdEnc setBuffer:mat->fwd            offset:0 atIndex:2];
			[cmdEnc dispatchThreads:spec.threadsPerGrid
			  threadsPerThreadgroup:spec.threadsPerThreadgroup];
			[cmdEnc endEncoding];
		} break;
		case Op_neg: {
			encode_update(cmdBuf, Mat_numel(mat->arg0), -1, mat->bwd, 1, tape[mat->arg0].bwd, 1);
		} break;
		case Op_add: {
			if (tape[mat->arg0].reqGrad) {
				encode_update(cmdBuf, Mat_numel(mat->arg0), 1, mat->bwd, 1, tape[mat->arg0].bwd, 1);
			}
			if (tape[mat->arg1].reqGrad) {
				encode_update(cmdBuf, Mat_numel(mat->arg1), 1, mat->bwd, 1, tape[mat->arg1].bwd, 1);
			}
		} break;
		case Op_mul: {
			assert(false);
		} break;
		case Op_pow: {
			// requires 2 particular functions.
			if (tape[mat->arg0].reqGrad) {
				// arg0->grad[i] += t->grad[i] * arg1->value[0] * powf(arg0->value[i], arg1->value[0]-1);
				GridSpec spec = getGridSpecElementwiseFun(powArg0BwdPSO, Mat_numel(mat->arg0));
				id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
				[cmdEnc setComputePipelineState:powArg0BwdPSO];
				[cmdEnc setBuffer:tape[mat->arg0].bwd offset:0 atIndex:0];
				[cmdEnc setBuffer:mat->bwd offset:0 atIndex:1];
				[cmdEnc setBytes:(float *)tape[mat->arg1].fwd.contents length:sizeof (float) atIndex:2];
				[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:3];
				[cmdEnc dispatchThreads:spec.threadsPerGrid
				  threadsPerThreadgroup:spec.threadsPerThreadgroup];
				[cmdEnc endEncoding];
			}
			if (tape[mat->arg1].reqGrad) {
				// arg1->grad[0] += t->grad[i] * t->value[i] * logf(arg0->value[i]);
				GridSpec spec = getGridSpecElementwiseFun(powArg1BwdPSO, Mat_numel(mat->arg0));
				id<MTLComputeCommandEncoder> cmdEnc = [cmdBuf computeCommandEncoder];
				[cmdEnc setComputePipelineState:powArg1BwdPSO];
				[cmdEnc setBytes:tape[mat->arg1].bwd.contents length:sizeof (float) atIndex:0];
				[cmdEnc setBuffer:mat->bwd offset:0 atIndex:1];
				[cmdEnc setBuffer:mat->fwd offset:0 atIndex:2];
				[cmdEnc setBuffer:tape[mat->arg0].fwd offset:0 atIndex:3];
				[cmdEnc dispatchThreads:spec.threadsPerGrid
				  threadsPerThreadgroup:spec.threadsPerThreadgroup];
				[cmdEnc endEncoding];
			}
		} break;
		case Op_dot: {
			if (tape[mat->arg0].reqGrad) {
				uint M = mat->shape.rows, N = tape[mat->arg1].shape.rows, K = mat->shape.cols;
				encode_matmul(cmdBuf, false, true, M, N, K,
							  mat->bwd, tape[mat->arg1].fwd, tape[mat->arg0].bwd);
			}
			if (tape[mat->arg1].reqGrad) {
				uint M = tape[mat->arg0].shape.cols, N = mat->shape.cols, K = tape[mat->arg0].shape.rows;
				encode_matmul(cmdBuf, true, false, M, N, K,
							  tape[mat->arg0].fwd, mat->bwd, tape[mat->arg1].bwd);
			}
		} break;
		}
	}

	commitWaitAndCheck(cmdBuf);
}

static void
enqueue_update_params(id<MTLCommandQueue> cmdQueue, float lr) {
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
	for (uint i = 0; i < op_count; i++) {
		if (!Mat_is_param(i)) {
			continue;
		}
		GridSpec spec = getGridSpecElementwiseFun(updatePSO, Mat_numel(i));
		MatRec *mat = tape+i;
		id<MTLComputeCommandEncoder> cmdEnc;
		cmdEnc = [cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
		[cmdEnc setComputePipelineState:updatePSO];
		[cmdEnc setBytes:&lr length:sizeof lr atIndex:0];
		[cmdEnc setBuffer:mat->bwd offset:0 atIndex:1];
		[cmdEnc setBytes:&(uint){1} length:sizeof(uint) atIndex:2];
		[cmdEnc setBuffer:mat->fwd offset:0 atIndex:3];
		[cmdEnc setBytes:&(uint){1} length:sizeof(uint) atIndex:4];
		[cmdEnc dispatchThreads:spec.threadsPerGrid
		  threadsPerThreadgroup:spec.threadsPerThreadgroup];
		[cmdEnc endEncoding];
	}
	
	commitWaitAndCheck(cmdBuf);
}

static void
enqueue_zero_grad(id<MTLCommandQueue> cmdQueue) {
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
	for (uint i = 0; i < op_count; i++) {
		if (!Mat_is_param(i)) {
			continue;
		}
		MatRec *mat = tape+i;
		id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
		[blitEnc fillBuffer:mat->bwd range:NSMakeRange(0, mat->bwd.length) value:0];
		[blitEnc endEncoding];
	}
	
	commitWaitAndCheck(cmdBuf);
}

static void
make_shared(Mat arg0) {
	// TODO: does this leak with ARC?
	MTLResourceOptions shared = MTLResourceStorageModeShared;
	MatRec *mat = tape+arg0;
	if (!(mat->fwd.resourceOptions & shared)) {
		mat->fwd = [dev newBufferWithLength:mat->fwd.length options:shared];
	}
	if (!(mat->bwd.resourceOptions & shared)) {
		mat->bwd = [dev newBufferWithLength:mat->bwd.length options:shared];
	}
}

void printMat(id<MTLCommandQueue> cmdQueue, Mat a) {
	MatRec *mat = tape+a;
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
	id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
	id<MTLBuffer> wtf = [dev newBufferWithLength:mat->fwd.length
										 options:MTLResourceStorageModeShared];
	[blitEnc copyFromBuffer:mat->fwd
			   sourceOffset:0
				   toBuffer:wtf
		  destinationOffset:0
					   size:wtf.length];
	[blitEnc endEncoding];
	[cmdBuf commit];
	[cmdBuf waitUntilCompleted];
	for (size_t i = 0; i < Mat_numel(a); i++) {
		NSLog(@"wtf: %f", ((float *)wtf.contents)[i]);
	}
}

// for debugging: https://melatonin.dev/blog/how-to-create-lldb-type-summaries-and-synthetic-children-for-your-custom-types/

int main() {
	NSError *error = nil;
	dev = MTLCreateSystemDefaultDevice();
	id<MTLLibrary> lib = [dev newDefaultLibrary];
	id<MTLCommandQueue> cmdQueue = [dev newCommandQueue];

	MTLFunctionConstantValues *consts = [MTLFunctionConstantValues new];
	for (int i = 1; i < Op_num; i++) {
		NSString *fwdName = [
			[NSString alloc] initWithFormat:@"%s%s", Op_str[i], "Fwd"
		];
		NSString *bwdName = [
			[NSString alloc] initWithFormat:@"%s%s", Op_str[i], "Bwd"
		];
		fwdFun[i] = [lib newFunctionWithName:fwdName
							  constantValues:consts
									   error:&error];
		fwdPSO[i] = [dev newComputePipelineStateWithFunction:fwdFun[i]
													   error:&error];
	}
	updateFun = [lib newFunctionWithName:@"update"];
	sigBwdFun = [lib newFunctionWithName:@"sigBwd"];
	powArg0BwdFun = [lib newFunctionWithName:@"powArg0Bwd"];
	powArg1BwdFun = [lib newFunctionWithName:@"powArg1Bwd"];

	updatePSO = [dev newComputePipelineStateWithFunction:updateFun error:&error];
	sigBwdPSO = [dev newComputePipelineStateWithFunction:sigBwdFun error:&error];
	powArg0BwdPSO = [dev newComputePipelineStateWithFunction:powArg0BwdFun error:&error];
	powArg1BwdPSO = [dev newComputePipelineStateWithFunction:powArg1BwdFun error:&error];

	// sum((sig(WX + B) - T)^2)
	// 0 a-4 = T
	// 1 a-3 = B
	// 2 a-2 = X
	// 3 a-1 = W
	// 4 a0 = WX
	// 5 a1 = a0 + B
	// 6 a2 = sig(a1)
	// 7 a3 = -T
	// 8 a4 = a2 + a3
	// 9 a5 = a4^2
	// 10 a6 = sum(a5)

	Mat W = Mat_new((MShape){4,3}, true);
	Mat X = Mat_new((MShape){3,4}, false);
	Mat B = Mat_new((MShape){4,4}, true);
	Mat T = Mat_new((MShape){4,4}, false);
	Mat E = Mat_new(scalar_shape, false);
	Mat Y = Mat_sig(Mat_add(Mat_dot(W, X), B)); // linear layer with sigmoid activation.
	Mat nT = Mat_neg(T);
	Mat YpnT = Mat_add(Y, nT);
	Mat YpnTe2 = Mat_pow(YpnT, E);
	Mat l = Mat_sum(YpnTe2); // mean squared error loss
	assert(l != 0);
	make_shared(l);

	for (int i = 0; i < 4*3; i++) {
		((float *)tape[W].fwd.contents)[i] = 1;
		((float *)tape[X].fwd.contents)[i] = 1;
	}
	for (int i = 0; i < 4*4; i++) {
		((float *)tape[B].fwd.contents)[i] = .5;
		((float *)tape[T].fwd.contents)[i] = 1;
	}
	*(float *)tape[E].fwd.contents = 2;

	// our objective for now is to overfit a batch (i.e. X, T).
	for (int i = 0; i < 10; i++) {
		// get batch (new X and T)
		enqueue_forward(cmdQueue);
		float loss = *(float *)tape[l].fwd.contents;
		NSLog(@"%f", loss);
		if (isnan(loss)) {
			printMat(cmdQueue, E);
			NSLog(@"");
			printMat(cmdQueue, YpnT);
			NSLog(@"");
			printMat(cmdQueue, YpnTe2);
			break;
		}
		enqueue_backward(cmdQueue);
		enqueue_update_params(cmdQueue, -0.005);
		enqueue_zero_grad(cmdQueue);
		// test for accuracy (turn off gradient traking)
	}

	return 0;
}

// https://alain.xyz/blog/raw-metal
// https://donaldpinckney.com/metal/2018/07/27/metal-intro-2.html
