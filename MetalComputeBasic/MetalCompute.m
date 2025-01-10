/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalCompute.h"

#include "tests.m"


// The number of floats in each array, and the size of the arrays in bytes.
//const unsigned int arrayLength = 1 << 24;
const unsigned int debugLength = 256;
const unsigned int debugSize = debugLength * sizeof(int);

const unsigned long arrayLength = 1000 * 256 * 256;
const unsigned long bufferSize = arrayLength * sizeof(float);
const unsigned long bufferSizeDot = arrayLength * sizeof(int);

const unsigned long matrixN = 1024; // 1 << 8 = 256
const unsigned long matrixLength = matrixN * matrixN; // 4 * 4 matrix
const unsigned long matrixSize = matrixLength * sizeof(float);

CFAbsoluteTime startTimeCPU;
CFAbsoluteTime endTimeCPU;

CFAbsoluteTime startTimeGPU;
CFAbsoluteTime endTimeGPU;

@implementation MetalCalc
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mAddFunctionPSO; // add function
    id<MTLComputePipelineState> _mSubFunctionPSO; // sub function
    id<MTLComputePipelineState> _mDotFunctionPSO; // dot function
    id<MTLComputePipelineState> _mMatMulFunctionPSO; // matmul function

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;
    
    id<MTLBuffer> arrayLengthBuffer;
    
    // Debug Buffer
    id<MTLBuffer> _mBufferDebug;

    // Buffers to hold data.
    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferResult;
    
    // integer buffers for dot product
    id<MTLBuffer> _mBufferADot;
    id<MTLBuffer> _mBufferBDot;
    id<MTLBuffer> _mBufferResultDot;
    
    // integer buffers for matmul product
    id<MTLBuffer> _mBufferAMatMul;
    id<MTLBuffer> _mBufferBMatMul;
    id<MTLBuffer> _mBufferResultMatMul;
    

}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;

        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil)
        {
            NSLog(@"Failed to find the default library.");
            return nil;
        }

        // MTL add function
        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"add_arrays"];
        if (addFunction == nil)
        {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }
        
        // MTL Sub function
        id<MTLFunction> subFunction = [defaultLibrary newFunctionWithName:@"sub_arrays"];
        if (subFunction == nil)
        {
            NSLog(@"Failed to find the sub function.");
            return nil;
        }
        
        // MTL Dot function
        id<MTLFunction> dotFunction = [defaultLibrary newFunctionWithName:@"dot_arrays"];
        if (dotFunction == nil)
        {
            NSLog(@"Failed to find the dot function.");
            return nil;
        }
        
        // MTL MatMul function
        id<MTLFunction> matMulFunction = [defaultLibrary newFunctionWithName:@"matmul_arrays"];
        if (matMulFunction == nil)
        {
            NSLog(@"Failed to find the matmul function.");
            return nil;
        }

        // Create a compute pipeline state object.
        // previous function is a proxy for MSL function, but is not executable code: need to convert function into executable code with a pipeline
        // PSO add function
        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction: addFunction error:&error];
        if (_mAddFunctionPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created Add pipeline state object, error %@.", error);
            return nil;
        }
        
        // PSO sub function
        _mSubFunctionPSO = [_mDevice newComputePipelineStateWithFunction: subFunction error:&error];
        if (_mSubFunctionPSO == nil)
        {
            NSLog(@"Failed to created Sub pipeline state object, error %@.", error);
            return nil;
        }
        
        // PSO dot function
        _mDotFunctionPSO = [_mDevice newComputePipelineStateWithFunction: dotFunction error:&error];
        if (_mDotFunctionPSO == nil)
        {
            NSLog(@"Failed to created Dot pipeline state object, error %@.", error);
            return nil;
        }
        
        // PSO MatMul function
        _mMatMulFunctionPSO = [_mDevice newComputePipelineStateWithFunction: matMulFunction error:&error];
        if (_mMatMulFunctionPSO == nil)
        {
            NSLog(@"Failed to created MatMul pipeline state object, error %@.", error);
            return nil;
        }

        // create COMMAND QUEUE that executes buffers
        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }

    return self;
}

- (void) prepareData
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // debug buffer
    _mBufferDebug = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    

    [self generateRandomFloatData:_mBufferA];
    [self generateRandomFloatData:_mBufferB];
}

- (void) prepareDataDot
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferADot = [_mDevice newBufferWithLength:bufferSizeDot options:MTLResourceStorageModeShared];
    _mBufferBDot = [_mDevice newBufferWithLength:bufferSizeDot options:MTLResourceStorageModeShared];
    _mBufferResultDot = [_mDevice  newBufferWithLength:bufferSizeDot options:MTLResourceStorageModeShared];
    
    // debug buffer
    _mBufferDebug = [_mDevice newBufferWithLength:debugSize options:MTLResourceStorageModeShared];
    
    

    [self generateRandomIntData:_mBufferADot];
    [self generateRandomIntData:_mBufferBDot];
}

- (void) prepareDataMatMul
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferAMatMul = [_mDevice newBufferWithLength:matrixSize options:MTLResourceStorageModeShared];
    _mBufferBMatMul = [_mDevice newBufferWithLength:matrixSize options:MTLResourceStorageModeShared];
    _mBufferResultMatMul = [_mDevice  newBufferWithLength:matrixSize options:MTLResourceStorageModeShared];
    
    // debug buffer
    _mBufferDebug = [_mDevice newBufferWithLength:debugSize options:MTLResourceStorageModeShared];
    

    [self generateRandomFloatDataMatMul:_mBufferAMatMul];
    [self generateRandomFloatDataMatMul:_mBufferBMatMul];
}


- (void) sendComputeCommand
{
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    
//////    // Start a compute pass.
//    id<MTLComputeCommandEncoder> computeEncoderAdd = [commandBuffer computeCommandEncoder];
//    assert(computeEncoderAdd != nil);
//
//    [self encodeAddCommand:computeEncoderAdd];
//    printf("done1\n");
////
//    // End the compute pass.
//    [computeEncoderAdd endEncoding];
//    
//    // start GPU time measure
//    startTimeGPU = CFAbsoluteTimeGetCurrent();
//    
////     Execute the command buffer.
//    [commandBuffer commit];
//
////     Wait for the GPU to complete the first task.
//    [commandBuffer waitUntilCompleted];
//    
//    // end GPU time measure
//    endTimeGPU = CFAbsoluteTimeGetCurrent();
//    
//    [self verifyAddResults];

//    id<MTLComputeCommandEncoder> computeEncoderSub = [commandBuffer computeCommandEncoder];
//    assert(computeEncoderSub != nil);
//    
//    [self encodeSubCommand:computeEncoderSub];
//    printf("done2\n");
//    //[self verifySubResults];
//    
//    [computeEncoderSub endEncoding];
    
    // DOT
//    id<MTLComputeCommandEncoder> computeEncoderDot = [commandBuffer computeCommandEncoder];
//    assert(computeEncoderDot != nil);
//    
//    [self encodeDotCommand:computeEncoderDot];
//    printf("(dot) Computed dot. \n");
//    
//    [computeEncoderDot endEncoding];
    
//    // MATMUL
    id<MTLComputeCommandEncoder> computeEncoderMatMul = [commandBuffer computeCommandEncoder];
    assert(computeEncoderMatMul != nil);
    
    [self encodeMatMulCommand:computeEncoderMatMul];
    printf("Encoded matmul. \n");
    
    [computeEncoderMatMul endEncoding];

    // start GPU time measure
    startTimeGPU = CFAbsoluteTimeGetCurrent();
    
    // Execute the command.
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];
    
    // end GPU time measure
    endTimeGPU = CFAbsoluteTimeGetCurrent();
    

//    [self verifyAddResults];
//    [self verifyDotResults];
    [self verifyMatMulResults];
}

- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    printf("add");
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];


    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
    
}

// adding "sub" function
- (void)encodeSubCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    printf("sub");
    [computeEncoder setComputePipelineState:_mSubFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mSubFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
    
}

// adding "dot" function
- (void)encodeDotCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    printf("dot\n");
    [computeEncoder setComputePipelineState:_mDotFunctionPSO];
    [computeEncoder setBuffer:_mBufferADot offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferBDot offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResultDot offset:0 atIndex:2]; // integer buffer, will become an atomic buffer
    
    // Pass arrayLength to the kernel
    [computeEncoder setBytes:&arrayLength length:sizeof(arrayLength) atIndex:3];
    
    // debug buffer
    [computeEncoder setBuffer:_mBufferDebug offset:0 atIndex:4]; // integer buffer, will become an atomic buffer
    
    // arrayLength

    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mDotFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    printf("Max Thread Group Size: %lu\n", threadGroupSize);
    
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

// adding "matmul" function
- (void)encodeMatMulCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    printf("matmul\n");
    [computeEncoder setComputePipelineState:_mMatMulFunctionPSO];
    [computeEncoder setBuffer:_mBufferAMatMul offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferBMatMul offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResultMatMul offset:0 atIndex:2]; // integer buffer, will become an atomic buffer
    
//    // Pass arrayLength to the kernel
//    [computeEncoder setBytes:&arrayLength length:sizeof(matrixLength) atIndex:3];
    
    // debug buffer
    [computeEncoder setBuffer:_mBufferDebug offset:0 atIndex:3]; // integer buffer, will become an atomic buffer
    
    // arrayLength

//    MTLSize gridSize = MTLSizeMake(matrixN, matrixN, 1); // covers the entire matrix
    // but I am using a flattened matrix (can't pass 2D array to Metal kernel)
    MTLSize gridSize = MTLSizeMake(matrixN, matrixN, 1); // covers the entire 1D FLATTENED matrix
    

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mMatMulFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > matrixN)
    {
        threadGroupSize = matrixN;
    }
    printf("Max Thread Group Size: %lu\n", threadGroupSize);
    
    MTLSize threadgroupSize = MTLSizeMake(sqrt(threadGroupSize), sqrt(threadGroupSize), 1); // root(n) * root(n) = n = threadGroupSize

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}


- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
//        dataPtr[index] = 1.123123981729387;
    }
}

- (void) generateRandomIntData: (id<MTLBuffer>) buffer
{
    int* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
//        dataPtr[index] = 1;
        dataPtr[index] = (rand() % 2);
//        dataPtr[index] = 1.123123981729387;
    }
}

- (void) generateRandomIntDataMatMul: (id<MTLBuffer>) buffer
{
    int* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < matrixLength; index++)
    {
//        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
//        dataPtr[index] = 1;
        dataPtr[index] = (rand() % 2);
//        dataPtr[index] = 1.123123981729387;
    }
}

- (void) generateRandomFloatDataMatMul: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < matrixLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
        
//        dataPtr[index] = 0.1;
//        dataPtr[index] = (rand() % 2);
//        dataPtr[index] = 1.123123981729387;
    }
}



- (void) verifyAddResults
{
    float* a = _mBufferA.contents;
    float* b = _mBufferB.contents;
    float* result = _mBufferResult.contents;
    

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (result[index] != (a[index] + b[index]))
        {
            printf("Add Compute ERROR: index=%lu result=%g vs %g=a+b\n",
                   index, result[index], a[index] + b[index]);
            assert(result[index] == (a[index] + b[index]));
        }
    }
    printf("(Adder) Compute results as expected\n");
}

- (void) verifySubResults
{
    float* a = _mBufferA.contents;
    float* b = _mBufferB.contents;
    float* result = _mBufferResult.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (result[index] != (a[index] - b[index]))
        {
            printf("Sub Compute ERROR: index=%lu result=%g vs %g=a-b\n",
                   index, result[index], a[index] - b[index]);
            assert(result[index] == (a[index] - b[index]));
        }
    }
    printf("(Sub) Compute results as expected\n");
}

- (void) verifyDotResults
{
    int* a = _mBufferADot.contents;
    int* b = _mBufferBDot.contents;
    int* result = _mBufferResultDot.contents;
    
    
//    int* debug =_mBufferDebug.contents;
//    // print debug buffer
//    for (int i = 0; i < debugLength; i++) {
//        printf("%u\n", debug[i]);
//    }
    
    int dot = 0;

    // CPU dot product computation; with timing
    CFAbsoluteTime startTimeCPU = CFAbsoluteTimeGetCurrent();
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dot += a[index] * b[index];
    }
    CFAbsoluteTime endTimeCPU = CFAbsoluteTimeGetCurrent();
    
    if (*result != dot)
    {
        printf("Dot Compute ERROR\n");
        printf("Dot: %u\n", dot);
        printf("Result: %u\n", *result);
//        assert(*result == dot);
    }
    
    CFAbsoluteTime timeCPU = endTimeCPU - startTimeCPU;
    CFAbsoluteTime timeGPU = endTimeGPU - startTimeGPU;
    CFAbsoluteTime speedup = timeCPU / timeGPU;
    
    
    printf("CPU compute time: %f\n", timeCPU);
    printf("GPU compute time: %f\n", timeGPU);
    printf("Speedup: %ux\n\n", (int) speedup);
    
    printf("Dot: %u\n", dot);
    printf("Result: %u\n", *result);
    printf("(Dot) Compute results as expected\n");
}

- (void) verifyMatMulResults
{
    float* a = _mBufferAMatMul.contents;
    float* b = _mBufferBMatMul.contents;
    float* result = _mBufferResultMatMul.contents;
    
    float reference[matrixLength];
    

    // CPU MatMul computation; with timing
    float acc = 0;
    CFAbsoluteTime startTimeCPU = CFAbsoluteTimeGetCurrent();
    for (unsigned long i = 0; i < matrixN; i++) {           // row
        for (unsigned long j = 0; j < matrixN; j++) {       // col
            acc = 0;
            for (unsigned long k = 0; k < matrixN; k++) {   //
                acc += a[(i * matrixN) + k] * b[j + (k * matrixN)];
            }
            reference[(i * matrixN) + j] = acc;
        }
    }
    CFAbsoluteTime endTimeCPU = CFAbsoluteTimeGetCurrent();
    
    
    // Check for error
    for (unsigned long i = 0; i < matrixLength; i++)
    {
//        printf("%f; %f\n", reference[i], result[i]);
        if (result[i] != reference[i])
        {
            printf("MatMul Compute ERROR: \n");
            assert(result[i] == reference[i]);
        }
    }
    
    // No compute error

    CFAbsoluteTime timeCPU = endTimeCPU - startTimeCPU;
    CFAbsoluteTime timeGPU = endTimeGPU - startTimeGPU;
    CFAbsoluteTime speedup = timeCPU / timeGPU;
    
    
    printf("CPU compute time: %f\n", timeCPU);
    printf("GPU compute time: %f\n", timeGPU);
    printf("Speedup: %ux\n\n", (int) speedup);

    printf("(MatMul) Compute results as expected\n");
}

@end
