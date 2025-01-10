# Performing Calculations on a GPU

Write Metal Shaders in MSL (Metal Shading Language) from scratch in xCode to run calculations on Apple M1 Pro Metal GPU.

_**Current Progress:**_
- 2x speedup on 1D Add, Sub, Prefix Sum, and Dot Product with metal kernel compared to CPU.
- 100x speedup on 2D float Matrix Multiplication on GPU compared to naive CPU implementation (when matrix length > 512).

**1D Prefix Sum Shader**
- Achieve **>2x speedup** compared to naive CPU implementation
- Use threadgroup shared "cache" for each threadgroup to compute the dot product synchronously, and have quicker shared memory writes across threads
- Use a binary reduction to reduce each threadgroup's cache data to the first position in the cache parallely
- Use an Atomic Add operation to accumulate the sum of all caches into the final result
  - Atomic operation is necessary to prevent race conditions on the final result, because all threadgroups should carefully access and write to RESULT
  - Metal only offers "int" atomic operations currently
- Follows parallel prefix sum algorithm: [https://en.wikipedia.org/wiki/Prefix_sum]
  
**1D Dot Product Shader**
- Achieve **>2x speedup** compared to naive CPU implementation
- Use threadgroup shared "cache" for each threadgroup to compute the dot product synchronously, and have quicker shared memory writes across threads
- First Attempt (10x slowdown): stride access pattern
- Second Attempt **(2x speedup)**: **continous access pattern** (on threadgroup works on one contigous part of data, not strided data)
- Adds each threadgroup's respective dot product using prefix sum algorithm
  - Binary Reduction
  - Atomic Add
 
**2D Matrix Multiplcation Shader**
- Achieve **>100x speedup** compared to naive O(N^3) CPU implementation
- Use 2D grid to cover entire result matrix; Use 2D threads in threadgroups
- Input matrices are flattened first (Metal shaders don't take 2D arrays as input; decided not to use Metal's "Matrix Class" to implement it from scratch)
- "Accumulator" stores each result position; no shared memory


Apple's Metal tutorial that inspired this project: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu.
