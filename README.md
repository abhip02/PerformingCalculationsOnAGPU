# Performing Calculations on a GPU

Write Metal Shaders in MSL (Metal Shading Language) in xCode to run calculations on Apple M1 Metal GPU.

_**Current Progress:**_
1D Add, Sub, and Dot Product achieves 2x speedup with metal kernel compared to CPU.
- Fixed "stride" memory access pattern for Dot Product: used continous accesses instead to achieve proper speedup with GPU parralelism

**Dot Product Kernel:**
- Use threadgroup shared "cache" for each threadgroup to compute the dot product synchronously, and have quicker shared memory writes across threads
- First Attempt (10x slowdown): stride access pattern
- Second Attempt **(2x speedup)**: **continous access pattern** (on threadgroup works on one contigous part of data, not strided data)
- Use a binary reduction to reduce each threadgroup's cache data to the first position in the cache parallely
- Use an Atomic Add operation to accumulate the sum of all caches into the final result
  - Atomic operation is necessary to prevent race conditions on the final result, because all threadgroups should carefully access and write to RESULT
  - Metal only offers "int" atomic operations currently
 

Apple's Metal tutorial that inspired this project: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu.
