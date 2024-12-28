/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A shader that adds two arrays of floats.
*/

#include <metal_stdlib>
using namespace metal;
kernel void sub_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] - inB[index];
}
