# SYCL-primitives : 2D Matrix Transpose
## 1. Overview  
2D matrix transpose operation with SYCL.  
The main optimization techniques are inspired by [1].  
It contains 4 versions for the transpose operation:  
- Naive implementation  
- Naive transpose via shared memory  
- Coalesced transpose via shared memory  
- Coalesced transpose via shared memory without shared memory bank conflict  

## 2. How to run
- mkdir build && cd build
- cmake ..
- make
- Run the executable 'transpose.out'
  
## 3. Implementation detail
- Naive implement
    - Basic parallel implementation for traspose. 
    - Each workitem at [i,j] copys the value at [i,j] into [j,i].
- Naive transpose via shared memory
    - Parallel implementation for traspose via shared memory. 
    - Each workitem at [i,j] copys the value at global [i,j] into local [li,lj].
    - Same workitem copys the value at local [li,lj] into global [j,i].
- Coalesced transpose via shared memory
    - Parallel implementation for traspose via shared memory with coalsced global memory update 
    - Each workitem at [i,j] in local group [I,J] copys the value at global [i,j] into local [li,lj].
    - Same workitem copys the value at local [li,lj] into global [i',j'], where i' = li+dI, j' = lj+dJ (d is local group size)
- Coalesced transpose via shared memory without shared memory bank conflict
## 4. Reference
[1] Mark Harris, An Efficient Matrix Transpose in CUDA C/C++, https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
