/*
Abstract: A shader that multiplies two matrices.
Assumptions:
 - A is flattened by row
 - B is flattened by col (otherwise would need strided accesses)
*/

#include <metal_stdlib>
using namespace metal;

// declare constant for MAX_THREADS per threadgroup; allows for constant compile-time build of "cache"
#define MAX_THREADS 256
#define matrixN 1024

[[max_total_threads_per_threadgroup(MAX_THREADS)]]
kernel void matmul_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       device int* debug,

                       uint2 index [[thread_position_in_grid]],
                       uint2 groupIndex [[threadgroup_position_in_grid]],
                       uint2 threadGroupIndex [[thread_position_in_threadgroup]],
                       uint2 threadgroupSize [[threads_per_threadgroup]])
{
    
//    // CPU MatMul computation; with timing
//    int acc = 0;
//    for (unsigned long i = 0; i < matrixN; i++) {           // row
//        for (unsigned long j = 0; j < matrixN; j++) {       // col
//            acc = 0;
//            for (unsigned long k = 0; k < matrixN; k++) {   //
//                acc += a[(i * matrixN) + k] * b[j + (k * matrixN)];
//            }
//            reference[(i * matrixN) + j] = acc;
//        }
//    }
    
    // 2D shared cache to store all "products"
    threadgroup float cache[MAX_THREADS]; // is matrixN = n length
//
    float sum = 0.0;
    // use threadgroup cache
    for (uint k = 0; k < matrixN; k++) {
//        cache[(threadGroupIndex.x * MAX_BUFFER_LENGTH) + threadGroupIndex.y] += inA[(index.x * matrixN) + k] * inB[index.y + (k * matrixN)];
        sum += inA[(index.x * matrixN) + k] * inB[index.y + (k * matrixN)];
    }

    // sync threads
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    
    result[(index.x * matrixN) + index.y] = sum;
}


//        [[1, 1],
//         [1, 1]]
//
//        [[1, 1],
//         [1, 1]]
//
//        Row 1 * col 1
//        Row 1 * col 2
//
//        Row 2 * col 1
//        Row 2 * col 1

// 8 total * operations
// 4 total + operations
