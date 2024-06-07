// xcrun -sdk macosx metal simple.metal
// clang traceme.m -framework Metal -framework CoreGraphics -framework Foundation -g -o traceme
// sudo dtruss -s traceme 2>&1

// This works only if in the directory where the executable is located there is a default.metallib

#import <Metal/Metal.h>

int main(void) {
	uint len = 1023;
	id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
	// Code loading.
	id<MTLLibrary> lib = [dev newDefaultLibrary];
	id<MTLFunction> fun = [lib newFunctionWithName:@"simple_function"];
	id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithFunction:fun
	                                                                     error:nil];
	// Data loading.
	id<MTLBuffer> buf = [dev newBufferWithLength:len
	                                     options:MTLResourceStorageModeShared];
	for (uint i = 0; i < len;  i++) ((float *)buf.contents)[i] = 1.f;
	// Work submission.
	id<MTLCommandQueue> cmdQueue = [dev newCommandQueue];
	id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
		id<MTLComputeCommandEncoder> cCmdEnc = [cmdBuf computeCommandEncoder];
		[cCmdEnc setComputePipelineState:cps];
		[cCmdEnc setBuffer:buf offset:0 atIndex:0];
		[cCmdEnc setBytes:&(float){1.0f} length:sizeof 1.0f atIndex:1];
		[cCmdEnc dispatchThreads:MTLSizeMake(len,1,1)
		   threadsPerThreadgroup:MTLSizeMake(cps.threadExecutionWidth,1,1)];
		[cCmdEnc endEncoding];
	[cmdBuf commit];
	[cmdBuf waitUntilCompleted];

	float *res = buf.contents;

	return 0;
}
