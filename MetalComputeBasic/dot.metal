/*
Abstract: A shader that dots two arrays.
*/

#include <metal_stdlib>
using namespace metal;

// declare constant for MAX_THREADS per threadgroup; allows for constant compile-time build of "cache"
#define MAX_THREADS 256

[[max_total_threads_per_threadgroup(MAX_THREADS)]]
kernel void dot_arrays(device const int* inA,
                       device const int* inB,
                       device _atomic<int>* result,
                       device int* debug,

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
    
    // local sum using STRIDE
    uint stride = threadgroupSize;
    uint forIndex = index;
    int temp = 0;
    while (forIndex < (1000 * 256 * 256)) {
        temp += inA[forIndex] * inB[forIndex];
        forIndex += stride;
    }
    
    cache[threadGroupIndex] = temp;

    // sync threads
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    
    // reduction (binary)
    stride = threadgroupSize/2;
    while (stride > 0) {
        if (threadGroupIndex < stride) {
            cache[threadGroupIndex] += cache[threadGroupIndex + stride];
        }
         //sync threads
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        stride /= 2;
    }

    //sync threads
   threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    
    // reduce first entry of each cache to result
    // only 1 thread should reduce, otherwise will overcount
    if (threadGroupIndex == 0 && groupIndex == 0) {
        atomic_fetch_add_explicit(result, cache[0], memory_order_relaxed);
    }
    
}
