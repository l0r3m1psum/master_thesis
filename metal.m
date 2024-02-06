#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <mach/mach_time.h>
#include "metal_common.h"

/* Per fare un po' di pratica con la programmazione GPU e la fusione di nuclei
 * mi serve scrivere un programma che legge un kernel metal da un file e
 * confronta il suo output con quello di una serie di operazioni (specificate da
 * riga di comando) implementate su CPU. In particolare mi serve la
 * moltiplicazione tra matrici, la moltiplicazione matrice vettore (da destra e
 * da sinistra), la somma (tra matrici e vettori), la sigmoide e la norma L2.
 * Idealmente i programmi implementati su GPU devono fare utilizzo della memoria
 * locale. La verifica che gli output sono gli stessi sarà fatta applicando i
 * due algoritmi a input casuali.
 * Come specifico la griglia di calcolo? Da riga di comando o la calcolo in
 * automatico in qualche modo?
 * Che dimensioni di input uso? Li prendo da riga di comando o li fisso ad un
 * numero grande, questo mi permetterebbe di monitorare le differenze di
 * prestazione tra CPU e GPU.
 * Devo anche considerare le differenze prestazionali tra nuclei fusi e no.
 * Quando testo una sequenza di operazioni devo provare la versione sequenziale
 * e fusa su GPU non solo su CPU (che serve solo per provare la correttezza
 * dell'implementazione).
 */

// https://github.com/YuAo/MetalLibraryArchive
// https://github.com/ShoYamanishi/AppleNumericalComputing/tree/main GOLD
// Given the fact that this GPU has threadgroup memory I guess it is not immediate mode
// https://developer.apple.com/forums/thread/668073

// good intro: https://developer.apple.com/videos/play/tech-talks/10580/
// good for performance https://developer.apple.com/videos/play/wwdc2020/10603/
// https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication?language=objc

// https://vgable.com/blog/2010/08/19/the-most-useful-objective-c-code-ive-ever-written/
// https://nshipster.com/nsassertionhandler/

// Shader validation is not supported on this platform. Enabling it disables
// capturing workloads. Also I have to set MTL_CAPTURE_ENABLED=1 as an
// environment variable.

// https://developer.apple.com/documentation/metal/mtlcomputepipelinereflection?language=objc

#define COUNT(a) (sizeof (a) / sizeof *(a))

typedef NS_ENUM(NSInteger, Error) {
	ErrorOk,
	ErrorInitialization,
	ErrorRuntime,
};

// https://www.gingerbill.org/article/2019/02/08/memory-allocation-strategies-002/
struct Arena {
	void   *base;
	size_t  used;
	size_t  size;
};

static bool
Arena_invariant(struct Arena * _Nonnull arena) {
	if (arena->base == NULL)       return false;
	if (arena->used > arena->size) return false;
	return true;
}

static void *
Arena_allocate(struct Arena * _Nonnull arena, size_t size) {
	NSCAssert(Arena_invariant(arena), @"Invariant violated");
	size_t allign = sizeof (void *)*4;
	// Because metal wants it this way.
	static_assert(sizeof (void *)*4 == 32, "wut?");

	// NOTE: should I check for overflow here? Ideally the Arena should be
	// initialized such that this can never happen.
	uintptr_t res = (uintptr_t)arena->base + (uintptr_t)arena->used;
	uintptr_t modulo = res % allign;
	if (modulo != 0) {
		res += allign - modulo;
	}
	NSCAssert(res%allign == 0, @"the pointer should be alligned by now!");

	if (res + size > (uintptr_t)(arena->base + arena->size)) {
		return NULL;
	}
	
	// This takes into account also the displacement caused by allignment.
	arena->used = res - (uintptr_t)arena->base + size;
	return res;
}

static ptrdiff_t
Arena_offset(struct Arena * _Nonnull arena, const void *p) {
	NSCAssert(Arena_invariant(arena), @"Invariant violated");
	// https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095
	if (!(   (uintptr_t)p >= (uintptr_t)arena->base
		  && (uintptr_t)p <  (uintptr_t)(arena->base + arena->used))) {
		NSCAssert(false, @"Pointer not in range");

	}
	return p - arena->base;
}

void generateRandomFloatData(float * _Nonnull dataPtr, NSUInteger arrayLength) {
	for (NSUInteger index = 0; index < arrayLength; index++) {
		dataPtr[index] = (float)rand()/(float)(RAND_MAX);
	}
}

static NSString *
NSStringFromMTLSize(MTLSize size) {
	return [NSString stringWithFormat:@"%tu %tu %tu", size.width, size.height, size.depth];
}

static BOOL
MTLCounterSetContainsCounter(id<MTLCounterSet> counterSet, MTLCommonCounter counter) {
	for (id<MTLCounter> counter in counterSet.counters) {
		if ([counter.name isEqualToString:counter.name]) {
			return YES;
		}
	}
	return NO;
}

NSString *kernelSource = @""
	"kernel void add_arrays(\n"
	"    device const float* inA,\n"
	"    device const float* inB,\n"
	"    device float* result,\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    threadgroup int zero = 0;\n"
	"    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
	"    result[index] = inA[index] + inB[index] + zero;\n"
	"}";

void MyExceptionHandler(NSException * _Nonnull exception) {
	@try {
		[exception raise];
	} @catch(id e) {
		[e raise];
	}
}

static id
check(id object, NSError *error, Error code, NSString *msg, ...) {
	NSCAssert(code != ErrorOk, @"The error code does not indicate an error.");
	// TODO: can I print the tipe of the object?
	if (error) {
		NSLog(@"%@: %@", error, msg);
		exit((int)code);
	}
	if (!object) {
		NSLog(@"%@", msg);
		exit((int)code);
	}
	return object;
}

static MTLCommandBufferHandler
startTimersAndGetTimeReportHandler(id<MTLDevice> device, id<MTLCounterSampleBuffer> counterSampleBuffer, NSRange counterRange) {
	MTLTimestamp cpuTimestampStart;
	MTLTimestamp gpuTimestampStart;
	[device sampleTimestamps:&cpuTimestampStart
				gpuTimestamp:&gpuTimestampStart];
	uint64_t rdtscStart = mach_absolute_time();

	MTLCommandBufferHandler res = ^(id<MTLCommandBuffer> _Nonnull commandBuffer) {
		NSLog(@"Reporting performance of %p %@", commandBuffer, commandBuffer.label);
		if (commandBuffer.status == MTLCommandBufferStatusError) {
			NSLog(@"runtime issue in the execution of the command buffer.");
			return;
		}
		MTLTimestamp cpuTimestampEnd = 0, gpuTimestampEnd = 0;
		[device sampleTimestamps:&cpuTimestampEnd gpuTimestamp:&gpuTimestampEnd];
		uint64_t rdtscEnd = mach_absolute_time();

		uint64_t rdtscNanoSec = 0;
		{
			// https://stackoverflow.com/a/23378064
			mach_timebase_info_data_t tbInfo;
			if (mach_timebase_info(&tbInfo) != KERN_SUCCESS) {
				NSLog(@"What the fuck (mach_timebase_info)?");
				return;
			}
			uint64_t rdtscElapsed = rdtscEnd - rdtscStart;

			uint64_t high = (rdtscElapsed >> 32) * tbInfo.numer;
			uint64_t low = (rdtscElapsed & 0xffffffffull) * tbInfo.numer / tbInfo.denom;
			uint64_t highRem = ((high % tbInfo.denom) << 32) / tbInfo.denom;
			high /= tbInfo.denom;
			rdtscNanoSec = (high << 32) + highRem + low;
		}


		// TODO: convert cycles to seconds
		NSLog(@"MTLCommandBuffer  scheduling time: %fs",  commandBuffer.kernelEndTime - commandBuffer.kernelStartTime);
		NSLog(@"MTLCommandBuffer     running time: %fs",  commandBuffer.GPUEndTime - commandBuffer.GPUStartTime);
		// TODO: This calculation should be abstracted in some way, especially the size of the matrix!
		NSLog(@"GFLOPS                   (matmul): %f", 2*pow(1024*1, 3)/(commandBuffer.GPUEndTime - commandBuffer.GPUStartTime)/10e9);
		NSLog(@"MTLDevice                CPU time: %llu", cpuTimestampEnd - cpuTimestampStart);
		NSLog(@"MTLDevice                GPU time: %llu", gpuTimestampEnd - gpuTimestampStart);
		NSLog(@"rdtsc                        time: %fs", rdtscNanoSec/(float)NSEC_PER_SEC);
		if (!counterSampleBuffer) {
			return;
		}
		NSData *data = [counterSampleBuffer resolveCounterRange:counterRange];
		if (!data) {
			NSLog(@"Unable to resolve sampled counters!");
			return;
		}
		MTLCounterResultTimestamp *sample = (MTLCounterResultTimestamp *)[data bytes];
		for (NSUInteger i = counterRange.location; i < counterRange.length; i++) {
			if (sample[i].timestamp == MTLCounterErrorValue) {
				NSLog(@"MTLCounterResultTimestamp[%lu] time: MTLCounterErrorValue", i);
				continue;
			}
			// https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/converting_gpu_timestamps_into_cpu_time?language=objc
			double normalizedGpuTime =
				(double)(sample[i].timestamp - gpuTimestampStart)
				/(gpuTimestampEnd - gpuTimestampStart);
			// This should be in nanoseconds but the printed time in seconds
			// makes no sense. The only thing that would make it make sense
			// would be femtoseconds... But it is so absurd
			double picoseconds = normalizedGpuTime * (cpuTimestampEnd - cpuTimestampStart)
				+ cpuTimestampStart;
			NSLog(@"MTLCounterResultTimestamp[%lu] time: %lfs", i, picoseconds/10e12);
		}
	};

	return Block_copy(res); // https://stackoverflow.com/a/15419698
}

static void
commitWaitAndCheck(id<MTLCommandBuffer> commandBuffer) {
	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];
	if (commandBuffer.error) {
		NSLog(@"%@", commandBuffer.error);
	}
}

// https://code.tutsplus.com/quick-tip-customize-nslog-for-easier-debugging--mobile-19066t

int main(int argc, const char **argv) {
	// FIXME: not all errors are cought???
	NSSetUncaughtExceptionHandler(MyExceptionHandler);
	NSError *error = nil;
	NSDateFormatter *dateFormatter = [NSDateFormatter new];
	[dateFormatter setDateFormat:@"HH:mm:ss.SSS"];

	NSLog(@"Starting initialization @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	check(device, error, ErrorInitialization, @"Failed to get the system's "
		  "default Metal device. Maybe you forgot to link with the CoreGraphics"
		  " framework https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc");
	id<MTLCommandQueue> commandQueue = [device newCommandQueue];
	check(commandQueue, error, ErrorInitialization, @"Failed to create the command queue.");
	NSUInteger maximumBufferSize = 1024ull*1024*3072;
	id<MTLBuffer> buffer = [device newBufferWithLength:maximumBufferSize
											   options:MTLResourceStorageModeShared];
	check(buffer, error, ErrorInitialization, @"unable to create buffer");

	struct Arena *arena = &(struct Arena){
		.base = buffer.contents, .used = 0, .size = maximumBufferSize
	};
#if 1
	NSBundle *bundle = [NSBundle mainBundle];
	check(bundle, error, ErrorInitialization, @"could not create bundle");
	id<MTLLibrary> library = [device newDefaultLibraryWithBundle:bundle error:&error];
#else
	id<MTLLibrary> library = [device newLibraryWithSource:kernelSource
												  options:nil
													error:&error];
#endif
	check(library, error, ErrorInitialization, @"Failed compile library");

	id<MTLCounterSet> counterSetTimestamp = nil, counterSetStageUtilization = nil;
	for (id<MTLCounterSet> counterSet in device.counterSets) {
		if        ([[counterSet name] isEqualToString:MTLCommonCounterSetTimestamp]) {
			counterSetTimestamp = counterSet;
		} else if ([[counterSet name] isEqualToString:MTLCommonCounterSetStageUtilization]) {
			counterSetStageUtilization = counterSet;
		} else if (!counterSetTimestamp & !counterSetStageUtilization) {
			break;
		}
	}
	if (!counterSetTimestamp | !counterSetStageUtilization) {
		NSLog(@"The counter sets we need is not present.");
		return ErrorInitialization;
	}
	// blit = block information transfer
	if (!([device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary]
		  & [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary])) {
		NSLog(@"The counter sampling point we need are not supported.");
		return ErrorInitialization;
	}
	if (!MTLCounterSetContainsCounter(counterSetTimestamp, MTLCommonCounterTimestamp)
		| !MTLCounterSetContainsCounter(counterSetStageUtilization, MTLCommonCounterTotalCycles)) {
		NSLog(@"the counters we need are not supported");
		return ErrorInitialization;
	}

	// NOTE: using the capture manager should be controlled by a command line argument.
	MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
	NSCAssert(captureManager, @"this should always exist?");
	if (!([captureManager supportsDestination:MTLCaptureDestinationGPUTraceDocument]
		  & [captureManager supportsDestination:MTLCaptureDestinationDeveloperTools])) {
		NSLog(@"capture manager does not support the required destinations, "
			  "maybe you forgot to set MTL_CAPTURE_ENABLED=1 in the environment");
		return ErrorInitialization;
	}

	if (!MPSSupportsMTLDevice(device)) {
		NSLog(@"this device does not support MPS");
		return ErrorInitialization;
	}

	if ([[[NSProcessInfo processInfo] arguments] containsObject:@"-inspection"]) {
		// https://stackoverflow.com/questions/4405006/nslog-printf-specifier-for-nsinteger
		NSLog(@"name:                                  %@", device.name);
		// We support only Metal 2 and not 3!
		NSLog(@"supportsFamily:Metal3                  %d", [device supportsFamily:MTLGPUFamilyMetal3]);
		NSLog(@"supportsFamily:Common3                 %d", [device supportsFamily:MTLGPUFamilyCommon3]);
		NSLog(@"architecture.name:                     %@", device.architecture.name);
		NSLog(@"maximumConcurrentCompilationTaskCount: %tu", device.maximumConcurrentCompilationTaskCount);
		NSLog(@"shouldMaximizeConcurrentCompilation:   %d", device.shouldMaximizeConcurrentCompilation);
		NSLog(@"maxThreadgroupMemoryLength:            %tu", device.maxThreadgroupMemoryLength);
		NSLog(@"maxThreadsPerThreadgroup:              %@", NSStringFromMTLSize(device.maxThreadsPerThreadgroup));
		NSLog(@"supportsFunctionPointers:              %d", device.supportsFunctionPointers);
		NSLog(@"currentAllocatedSize:                  %tu", device.currentAllocatedSize);
		NSLog(@"recommendedMaxWorkingSetSize:          %llu", device.recommendedMaxWorkingSetSize);
		NSLog(@"hasUnifiedMemory:                      %d", device.hasUnifiedMemory);
		NSLog(@"maxTransferRate:                       %llu", device.maxTransferRate);
		NSLog(@"argumentBuffersSupport:                %d", device.argumentBuffersSupport);
		for (id<MTLCounterSet> counterSet in device.counterSets) {
			NSLog(@"%@", counterSet);
			for (id<MTLCounter> counter in counterSet.counters) {
				NSLog(@"\t%@", counter);
			}
		}
	}

	NSLog(@"Finished initialization @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	/********************************************************************************************************************/

	// TODO: what to do about allocation errors (when calling alloc, new or alloc)?

	NSUInteger loopCount = 10;
	MTLCounterSampleBufferDescriptor *counterSampleBufferDescriptor = [MTLCounterSampleBufferDescriptor new];
	counterSampleBufferDescriptor.sampleCount = loopCount;
	counterSampleBufferDescriptor.storageMode = MTLStorageModeShared;
	counterSampleBufferDescriptor.label       = @"idk a label that i will change at some point";
	counterSampleBufferDescriptor.counterSet  = counterSetTimestamp;
	id<MTLCounterSampleBuffer> sampleBuffer = [device newCounterSampleBufferWithDescriptor:counterSampleBufferDescriptor
																					 error:&error];
	check(sampleBuffer, error, ErrorRuntime, @"Unable to create sample buffer");

	counterSampleBufferDescriptor.sampleCount = [counterSetStageUtilization.counters count];
	counterSampleBufferDescriptor.counterSet = counterSetStageUtilization;
	// TODO: add sampleBuffer2 to the report
	id<MTLCounterSampleBuffer> sampleBuffer2 = [device newCounterSampleBufferWithDescriptor:counterSampleBufferDescriptor
																					  error:&error];
	check(sampleBuffer2, error, ErrorRuntime, @"Unable to create sample buffer 2");

	MTLCaptureDescriptor *captureDescriptor = [MTLCaptureDescriptor new];
	{
		captureDescriptor.captureObject = device;
		captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;
		NSFileManager *fileManager = [NSFileManager defaultManager];
		NSString *outputURL = [fileManager currentDirectoryPath];
		captureDescriptor.outputURL = [[[NSURL alloc] initFileURLWithPath:outputURL]
									   URLByAppendingPathComponent:@"trace.gputrace"];
		[fileManager removeItemAtURL:captureDescriptor.outputURL error:&error];
		error = nil;
	}
	if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
		NSLog(@"Unable to capture: %@", error);
		error = nil;
	}

#define WHICH_EXPERIMENT 3
#define WHICH_FUNC 1

#if WHICH_FUNC == 0
	NSUInteger
		matrixRows = 1021*1, matrixCols = matrixRows,
		matrixElems = matrixRows * matrixCols,
		matrixSize = sizeof (float) * matrixElems;
	uint *num_elem = Arena_allocate(arena, sizeof *num_elem);
	float
		*A          = Arena_allocate(arena, matrixSize),
		*B          = Arena_allocate(arena, matrixSize),
		*customC    = Arena_allocate(arena, matrixSize*10), // TODO: this will allow to run multiple kernel un parallel
		*referenceC = Arena_allocate(arena, matrixSize);
	NSCAssert(referenceC != NULL, @"the matrix are too big");
	*num_elem = matrixRows;
	generateRandomFloatData(A, matrixElems);
	generateRandomFloatData(B, matrixElems);

	// Populating the reference buffer.
	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"unable to create command buffer");
		commandBuffer.label = @"MPS reference";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, nil, NSMakeRange(0, 1))];
		size_t stride = [MPSMatrixDescriptor rowBytesForColumns:matrixRows
													   dataType:MPSDataTypeFloat32];
		stride = matrixRows*sizeof(float);
		MPSMatrixDescriptor *matrixDescriptor;
		matrixDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:matrixRows
																 columns:matrixCols
																rowBytes:stride
																dataType:MPSDataTypeFloat32];
		MPSMatrix
			*A_ = [[MPSMatrix alloc] initWithBuffer:buffer
											offset:Arena_offset(arena, A)
										descriptor:matrixDescriptor],
			*B_ = [[MPSMatrix alloc] initWithBuffer:buffer
											offset:Arena_offset(arena, B)
										descriptor:matrixDescriptor],
			*C_ = [[MPSMatrix alloc] initWithBuffer:buffer
											offset:Arena_offset(arena, referenceC)
										descriptor:matrixDescriptor];
		MPSMatrixMultiplication *matmul = [MPSMatrixMultiplication alloc];
		[matmul initWithDevice:device
					resultRows:matrixRows
				 resultColumns:matrixCols
			   interiorColumns:matrixCols];
		[matmul encodeToCommandBuffer:commandBuffer
						   leftMatrix:A_
						  rightMatrix:B_
						 resultMatrix:C_];

		commitWaitAndCheck(commandBuffer);
	}

#if WHICH_EXPERIMENT == 0
	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	id<MTLFunction> function = [library newFunctionWithName:@"matmul_naive"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");

	NSLog(@"Starting to issue commands @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
		commandBuffer.label = @"naive matmul";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];

		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

		[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
		[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, A) atIndex:0];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, B) atIndex:1];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, customC) atIndex:2];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, num_elem) atIndex:3];
		[computeEncoder dispatchThreadgroups:MTLSizeMake(32, 32, 1)
					   threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
		[computeEncoder endEncoding];

		commitWaitAndCheck(commandBuffer);
	}

	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"unable to create command buffer");
		commandBuffer.label = @"naive matmul ten times";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, loopCount))];
		NSLog(@"If you keep feeding the beast scheduling times go down. Also if"
			  "you execute them in a loop they get sliglhtly faster. Probably "
			  "due to cacheing.");

		for (NSUInteger i = 0; i < loopCount; i++) {
			id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
			check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

			[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:i withBarrier:YES];
			// [computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:i withBarrier:YES];
			[computeEncoder setComputePipelineState:functionPSO];
			[computeEncoder setBuffer:buffer offset:Arena_offset(arena, A) atIndex:0];
			[computeEncoder setBuffer:buffer offset:Arena_offset(arena, B) atIndex:1];
			[computeEncoder setBuffer:buffer offset:Arena_offset(arena, customC) atIndex:2];
			[computeEncoder setBuffer:buffer offset:Arena_offset(arena, num_elem) atIndex:3];
	#if 0
			[computeEncoder dispatchThreadgroups:MTLSizeMake(32, 32, 1)
						   threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
	#else
			[computeEncoder dispatchThreadgroups:MTLSizeMake(16, 64, 1)
						   threadsPerThreadgroup:MTLSizeMake(64, 16, 1)];
	#endif
			[computeEncoder endEncoding];
		}

		commitWaitAndCheck(commandBuffer);
	}

	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"unable to create command buffer");
		commandBuffer.label = @"MPS reference ten times";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, nil, NSMakeRange(0, 1))];

		for (int i = 0; i < 10; i++) {
			size_t stride = [MPSMatrixDescriptor rowBytesForColumns:matrixRows
														   dataType:MPSDataTypeFloat32];
			MPSMatrixDescriptor *matrixDescriptor;
			matrixDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:matrixRows
																	 columns:matrixCols
																	rowBytes:stride
																	dataType:MPSDataTypeFloat32];
			MPSMatrix
				*A_ = [[MPSMatrix alloc] initWithBuffer:buffer
												offset:Arena_offset(arena, A)
											descriptor:matrixDescriptor],
				*B_ = [[MPSMatrix alloc] initWithBuffer:buffer
												offset:Arena_offset(arena, B)
											descriptor:matrixDescriptor],
				*C_ = [[MPSMatrix alloc] initWithBuffer:buffer
												offset:Arena_offset(arena, referenceC)
											descriptor:matrixDescriptor];
			MPSMatrixMultiplication *matmul = [MPSMatrixMultiplication alloc];
			[matmul initWithDevice:device
						resultRows:matrixRows
					 resultColumns:matrixCols
				   interiorColumns:matrixCols];
			[matmul encodeToCommandBuffer:commandBuffer
							   leftMatrix:A_
							  rightMatrix:B_
							 resultMatrix:C_];
		}

		commitWaitAndCheck(commandBuffer);
	}

#elif WHICH_EXPERIMENT == 1

	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	id<MTLFunction> function = [library newFunctionWithName:@"matmul_naive"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");

	NSLog(@"Starting to issue commands @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	MTLSize sizes[] = {
		// MTLSizeMake(1, 16, 1),
		MTLSizeMake(1, 32, 1),
		MTLSizeMake(1, 64, 1),
		MTLSizeMake(1, 128, 1),
		MTLSizeMake(1, 256, 1),
		MTLSizeMake(1, 512, 1),

		MTLSizeMake(8, 8, 1),

		MTLSizeMake(16, 16, 1),
		MTLSizeMake(16, 32, 1),
		MTLSizeMake(32, 2, 1),
		MTLSizeMake(32, 16, 1),

		MTLSizeMake(32, 1, 1),
		MTLSizeMake(64, 1, 1),
		MTLSizeMake(128, 1, 1),
		MTLSizeMake(256, 1, 1),
		MTLSizeMake(512, 1, 1),
	};

	NSLog(@"The warp/wavefront size is really 64!");
	/*
	 * MPS reference
	 * scheduling time: 10.871407s
	 * running time: 0.024401s
	 * 1 32 1
	 * scheduling time: 1.572611s
	 * running time: 0.127838s
	 * 1 64 1
	 * scheduling time: 1.665154s
	 * running time: 0.077491s
	 * 1 128 1
	 * scheduling time: 1.558397s
	 * running time: 0.073786s
	 * 1 256 1
	 * scheduling time: 1.608655s
	 * running time: 0.072698s
	 * 1 512 1
	 * scheduling time: 1.578581s
	 * running time: 0.073441s
	 * 8 8 1
	 * scheduling time: 1.615669s
	 * running time: 0.148500s
	 * 16 16 1
	 * scheduling time: 1.606292s
	 * running time: 0.137394s
	 * 16 32 1
	 * scheduling time: 1.689098s
	 * running time: 0.136236s
	 * 32 2 1
	 * scheduling time: 1.580925s
	 * running time: 0.585094s
	 * 32 16 1
	 * scheduling time: 1.663665s
	 * running time: 0.217367s
	 * 32 1 1
	 * scheduling time: 1.588813s
	 * running time: 1.114747s
	 * 64 1 1
	 * scheduling time: 1.528778s
	 * running time: 1.873308s
	 * 128 1 1
	 * scheduling time: 1.599647s
	 * running time: 1.344569s
	 * 256 1 1
	 * scheduling time: 1.729080s
	 * running time: 1.529463s
	 * 512 1 1
	 * scheduling time: 1.638301s
	 * running time: 1.593372s
	 */

	for (size_t i = 0; i < COUNT(sizes); i++) {
		MTLSize size = sizes[i];
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
		commandBuffer.label = NSStringFromMTLSize(size);
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];
		[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
		[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, A) atIndex:0];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, B) atIndex:1];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, customC) atIndex:2];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, num_elem) atIndex:3];

		MTLSize grid = MTLSizeMake(matrixRows/size.width, matrixCols/size.height, 1),
			block = MTLSizeMake(size.width, size.height, 1);
		[computeEncoder dispatchThreadgroups:grid
					   threadsPerThreadgroup:block];
		[computeEncoder endEncoding];

		commitWaitAndCheck(commandBuffer);
		if (memcmp(customC, referenceC, matrixSize) != 0) {
			NSLog(@"custom and reference implementation disagree!");
			return ErrorRuntime;
		}
	}

#elif WHICH_EXPERIMENT == 2

	// TODO: implement rectangular tiles (to find the best block size).
	uint tile_width = 32;
	static_assert(sizeof tile_width == 4, "wut?");

	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	[functionConstantValues setConstantValue:&tile_width type:MTLDataTypeUInt atIndex:0];
	[functionConstantValues setConstantValue:&(bool){true} type:MTLDataTypeBool atIndex:1];
	int *debug = Arena_allocate(arena, matrixElems*sizeof *debug);
	id<MTLFunction> function = [library newFunctionWithName:@"matmul_tiled"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");

	NSLog(@"Starting to issue commands @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
	check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
	commandBuffer.label = @"Tiled matrix multiplication";
	id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
	check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

	[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];
	[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
	[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
	[computeEncoder setComputePipelineState:functionPSO];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, A) atIndex:0];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, B) atIndex:1];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, customC) atIndex:2];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, num_elem) atIndex:3];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, debug) atIndex:4];
	NSUInteger threadGroupLength = MAX(tile_width*tile_width*sizeof(float), 16);
	[computeEncoder setThreadgroupMemoryLength:threadGroupLength atIndex:0];
	[computeEncoder setThreadgroupMemoryLength:threadGroupLength atIndex:1];
	MTLSize grid = MTLSizeMake(CEIL_DIV(matrixRows, tile_width), CEIL_DIV(matrixCols, tile_width), 1),
		block = MTLSizeMake(tile_width, tile_width, 1);
	[computeEncoder dispatchThreadgroups:grid
				   threadsPerThreadgroup:block];
	[computeEncoder endEncoding];

	commitWaitAndCheck(commandBuffer);
#elif WHICH_EXPERIMENT == 3

	uint tile_width = 32;
	static_assert(sizeof tile_width == 4, "wut?");

	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	[functionConstantValues setConstantValue:&tile_width type:MTLDataTypeUInt atIndex:0];
	id<MTLFunction> function = [library newFunctionWithName:@"matmul_tiled_coarsed"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");
	NSLog(@"maxTotalThreadsPerThreadgroup: %zu", functionPSO.maxTotalThreadsPerThreadgroup);
	NSLog(@"threadExecutionWidth: %zu", functionPSO.threadExecutionWidth);
	NSLog(@"staticThreadgroupMemoryLength: %zu", functionPSO.staticThreadgroupMemoryLength);

	NSLog(@"Starting to issue commands @ %@", [dateFormatter stringFromDate:[NSDate now]]);

	id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
	check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
	commandBuffer.label = @"Tiled Coarsed matrix multiplication";
	id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
	check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

	[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];
	[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
	[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
	[computeEncoder setComputePipelineState:functionPSO];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, A) atIndex:0];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, B) atIndex:1];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, customC) atIndex:2];
	[computeEncoder setBuffer:buffer offset:Arena_offset(arena, num_elem) atIndex:3];
	NSUInteger threadGroupLength = MAX(tile_width*tile_width*sizeof(float), 16);
	[computeEncoder setThreadgroupMemoryLength:threadGroupLength atIndex:0];
	[computeEncoder setThreadgroupMemoryLength:threadGroupLength atIndex:1];
	MTLSize grid = MTLSizeMake(CEIL_DIV(matrixRows, tile_width)/COARSE_FACTOR, CEIL_DIV(matrixCols, tile_width), 1),
		block = MTLSizeMake(tile_width, tile_width, 1);
	[computeEncoder dispatchThreadgroups:grid
				   threadsPerThreadgroup:block];
	[computeEncoder endEncoding];

	commitWaitAndCheck(commandBuffer);
#endif

	// FIXME: do we have an allignment problem with MPS??

	[captureManager stopCapture];

	float mae = 0;
	bool printed = false;
	for (size_t i = 0; i < matrixElems; i++) {
		mae += fabsf(referenceC[i] - customC[i]);
		if (mae != 0 && !printed) {
			NSLog(@"%zu", i);
			printed = true;
		}
	}
	NSLog(@"mean absolute error: %f", mae);

	for (size_t i = 0; i < matrixElems; i++) {
		if (referenceC[i] != customC[i]) {
			printf("%zu ", i);
		}
	}

	if (memcmp(customC, referenceC, matrixSize) != 0) {
		NSLog(@"custom and reference implementation disagree!");
		return ErrorRuntime;
	}
#else

	NSUInteger arrayLen = 1024*4, arraySize = arrayLen * sizeof(float);
	float
		*array = Arena_allocate(arena, arraySize),
		*sumCustom = Arena_allocate(arena, sizeof *sumCustom);
	NSCAssert(sumCustom != NULL, @"the array is too big");
	generateRandomFloatData(array, arrayLen);

	float sumReference = 0;
	for (size_t i = 0; i < arrayLen; i++) {
		sumReference += array[i];
	}

#if WHICH_EXPERIMENT == 0
	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	id<MTLFunction> function = [library newFunctionWithName:@"sum"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");

	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
		commandBuffer.label = @"naive sum";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];

		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

		[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
		[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, array) atIndex:0];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, sumCustom) atIndex:1];
		[computeEncoder setThreadgroupMemoryLength:arrayLen/2*sizeof(float) atIndex:0];
		[computeEncoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
					   threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
		[computeEncoder endEncoding];

		commitWaitAndCheck(commandBuffer);
	}

#else

	MTLFunctionConstantValues *functionConstantValues = [MTLFunctionConstantValues new];
	id<MTLFunction> function = [library newFunctionWithName:@"sum_coarsed"
											 constantValues:functionConstantValues
													  error:&error];
	check(function, error, ErrorRuntime, @"Failed to find the function");
	id<MTLComputePipelineState> functionPSO = [device newComputePipelineStateWithFunction:function
																					error:&error];
	check(functionPSO, error, ErrorRuntime, @"Failed to created pipeline state object");

	{
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		check(commandBuffer, error, ErrorRuntime, @"Failed to create the command buffer.");
		commandBuffer.label = @"naive sum";
		[commandBuffer addCompletedHandler:startTimersAndGetTimeReportHandler(device, sampleBuffer, NSMakeRange(0, 1))];

		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		check(computeEncoder, error, ErrorRuntime, @"Failed to create the compute encoder.");

		[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:YES];
		[computeEncoder sampleCountersInBuffer:sampleBuffer2 atSampleIndex:0 withBarrier:YES];
		[computeEncoder setComputePipelineState:functionPSO];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, array) atIndex:0];
		[computeEncoder setBuffer:buffer offset:Arena_offset(arena, sumCustom) atIndex:1];
		[computeEncoder setThreadgroupMemoryLength:arrayLen/2*sizeof(float) atIndex:0];
		[computeEncoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
					   threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
		[computeEncoder endEncoding];

		commitWaitAndCheck(commandBuffer);
	}

#endif

	float err = fabs(sumReference - *sumCustom), epsilon = 0.01;
	if (err > epsilon) {
		return  ErrorRuntime;
	}
#endif

	return ErrorOk;
}

// https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores
// http://yosefk.com/blog/simd-simt-smt-parallelism-in-nvidia-gpus.html wow amazing blog post.

// https://stackoverflow.com/questions/50826644/why-do-we-do-batch-matrix-matrix-product
// https://datascience.stackexchange.com/questions/66913/how-does-attention-mechanism-learn

// https://developer.apple.com/forums/thread/46817
