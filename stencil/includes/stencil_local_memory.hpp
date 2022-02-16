#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/sycl.hpp>
namespace sycl=cl::sycl;

template<int K_SIZE>
void stencil_local_memory(sycl::queue queue, DTYPE* in, DTYPE* kernel, DTYPE* out) {

    const int K_HALF=K_SIZE/2;


    queue.submit([&] (sycl::handler& cgh){

        sycl::accessor<DTYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> local_kernel(sycl::range<2>(K_SIZE,K_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<2>({N,N}, {16, 16}), [=](sycl::nd_item<2> item) {

            int x = item.get_global_id(0);
            int y = item.get_global_id(1);
            int local_x = item.get_local_id(0);
            int local_y = item.get_local_id(1);

            int ky, kx;

            if (local_x<K_SIZE && local_y<K_SIZE) {
                local_kernel[local_y][local_x] = kernel[(local_y*K_SIZE)+local_x];
            }
            item.barrier(sycl::access::fence_space::local_space);        

            DTYPE sum=0;            
            for (ky=-K_HALF; ky<=K_HALF; ky++) {
                for (kx=-K_HALF; kx<=K_HALF; kx++) {
                    if (0<=x+kx && x+kx<N && 0<=y+ky && y+ky<N) {
                        sum += in[(y+ky)*N+x+kx] * local_kernel[(ky+K_HALF)][(kx+K_HALF)];
                    }
                }
            }

            out[y*N+x] = sum;
        });
    });

    queue.wait();
}

