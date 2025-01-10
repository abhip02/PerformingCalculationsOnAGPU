/*
Abstract: A shader that computes (prefix) sum of array.
 - Shared threadgroup cache
 - Uses atomic add at end to write to "result"
*/

#include <metal_stdlib>
using namespace metal;

// declare constant for MAX_THREADS per threadgroup; allows for constant compile-time build of "cache"
#define MAX_THREADS 256

[[max_total_threads_per_threadgroup(MAX_THREADS)]]
kernel void prefix_sum(device const int* inA,
                       device _atomic<int>* result,

                       uint index [[thread_position_in_grid]],
                       uint groupIndex [[threadgroup_position_in_grid]],
                       uint threadGroupIndex [[thread_position_in_threadgroup]],
                       uint threadgroupSize [[threads_per_threadgroup]])
{
    
    // Threads in threadgroup shared memory = "shared" buffer
    threadgroup float cache[MAX_THREADS];
    
    int temp = 0;
    temp += inA[index];
    cache[threadGroupIndex] = temp;
    
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
