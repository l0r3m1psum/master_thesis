#import <Metal/Metal.h>

NSString *kernelSource = @""
	"kernel void add_arrays(\n"
	"    device const float *inA   [[buffer(0)]],\n"
	"    device const float *inB   [[buffer(1)]],\n"
	"    device       float *outC  [[buffer(2)]],\n"
	"                 uint   index [[thread_position_in_grid]]\n"
	") {\n"
	"    outC[index] = inA[index] + inB[index];\n"
	"}";

int main(int argc, const char **argv) {
	NSError *error = nil;

	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	id<MTLCommandQueue> commandQueue = [device newCommandQueue];
	NSUInteger maximumBufferSize = 1024ull*1024*3072;
	id<MTLBuffer> buffer = [device newBufferWithLength:maximumBufferSize
											   options:MTLResourceStorageModeShared];
	id<MTLLibrary> library = [device newLibraryWithSource:kernelSource
												  options:nil
													error: &error];
	if (error != nil) {
		NSLog(@"Failed compile library %@.", error);
		return 1;
	}
	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"
												constantValues:functionConstantValues
														 error:&error];
	if (error != nil) {
		NSLog(@"Failed to find the adder function %@.", error);
		return 1;
	}
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	if (error != nil) {
		NSLog(@"Failed to created pipeline state object, error %@.", error);
		return 1;
	}

	id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

	{
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:0    atIndex:0];
		[computeEncoder setBuffer:buffer offset:1024 atIndex:1];
		[computeEncoder setBuffer:buffer offset:2048 atIndex:2];
		[computeEncoder dispatchThreads:MTLSizeMake(32, 1, 1)
				  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
		[computeEncoder endEncoding];
	}

	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];

	commandBuffer = [commandQueue commandBuffer];

	{
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:2048 atIndex:0];
		[computeEncoder setBuffer:buffer offset:1024 atIndex:1];
		[computeEncoder setBuffer:buffer offset:0    atIndex:2];
		[computeEncoder dispatchThreads:MTLSizeMake(32, 1, 1)
				  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
		[computeEncoder endEncoding];
	}

	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];

	return 0;
}


