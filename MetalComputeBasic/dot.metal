/*
Abstract: A shader that dots two arrays.
 - Shared threadgroup cache
 - Uses atomic add at end to write to "result"
*/

#include <metal_stdlib>
using namespace metal;

// declare constant for MAX_THREADS per threadgroup; allows for constant compile-time build of "cache"
#define MAX_THREADS 256

[[max_total_threads_per_threadgroup(MAX_THREADS)]]
kernel void dot_arrays(device const int* inA,
                       device const int* inB,
                       device _atomic<int>* result,
                       constant uint& arrayLength, // Pass arrayLength as a constant
                       device int* debug,
//                       device int* arrayLength,

                       uint index [[thread_position_in_grid]],
                       uint groupIndex [[threadgroup_position_in_grid]],
                       uint threadGroupIndex [[thread_position_in_threadgroup]],
                       uint threadgroupSize [[threads_per_threadgroup]])
{
    //    // Debugging: Write group-level information to debug buffer
    //    if (threadGroupIndex == 0) {  // Only one thread per group writes
    //        debug[groupIndex] = 1;  // Mark this group as active
    //    }
    // with 512 buffer size, I see 2 active groups (good)
    
    
    // Threads in threadgroup shared memory = "shared" buffer
    threadgroup float cache[MAX_THREADS];
    
    // local using partitions
    uint partitionLength = threadgroupSize;
    uint startIndex = groupIndex * partitionLength;
    //    uint endIndex = startIndex + partitionLength;
    
    int temp = 0;
    
    // this for loop is replaced with the one line below
    //    for (uint i = startIndex; i < endIndex; i++) {
    //        temp += inA[i] * inB[i];
    //    }
    
    // individual threads execute this parallelly (the for loop above)
//    temp += inA[startIndex + threadGroupIndex] * inB[startIndex + threadGroupIndex];
    temp += inA[index] * inB[index];
    cache[threadGroupIndex] = temp;
    
// local sum using STRIDE
// this method has inefficient memory access patterns:
    
//        uint stride = threadgroupSize;
//        uint forIndex = index;
//        int temp = 0;
//        while (forIndex < arrayLength) {
//            temp += inA[forIndex] * inB[forIndex];
//            forIndex += stride;
//        }
//    
//        cache[threadGroupIndex] = temp;
//  end stride method
    
    // sync threads
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    
    // reduction (binary)
    uint stride = threadgroupSize/2;
    while (stride > 0) {
        if (threadGroupIndex < stride) {
            cache[threadGroupIndex] += cache[threadGroupIndex + stride];
        }
        //sync threads
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        stride /= 2;
    }
    
    // accumulate all threadgroup caches
    // only 1 thread from each threadgroup should reduce, otherwise will overcount
    if (threadGroupIndex == 0) { // if stride accesses are used, add "&& groupIndex == 0"
        atomic_fetch_add_explicit(result, cache[0], memory_order_relaxed);
    }
}
