# SYCL-primitives : 2D Matrix Transpose
## 1. Overview  
2D matrix transpose operation by SYCL  
The main optimization techniques are inspired by [1]
It contains 4 versions for the transpose operation
    - Naive implement
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
    - Basic
- Naive transpose via shared memory
- Coalesced transpose via shared memory
- Coalesced transpose via shared memory without shared memory bank conflict
## 4. Reference
[1] Mark Harris, An Efficient Matrix Transpose in CUDA C/C++, https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/