#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/sycl.hpp>
namespace sycl=cl::sycl;

/*** Measure performance ***/
#include <sys/time.h>
#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))
timeval start, end;

/*** Data configuration ***/
#define DTYPE long
constexpr int N=1024;
constexpr int KERNEL_SIZE=3;

/*** Debugging info ***/
#define __MODE_DEBUG_TIME__
const int NUM_TESTS=20;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&,const std::vector<DTYPE>&);

/*** Parallel algorithm implementations ***/
#include "includes/stencil_naive.hpp"



int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel 2D Matrix Stencil\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 2D Matrix : in["<<N<<","<<N<<"] with kernel["<<KERNEL_SIZE<<","<<KERNEL_SIZE<<"] and zero padding -> "<<"out["<<N<<","<<N<<"]\n";
    std::cout << "-- 2D Matrix size: "<<sizeof(DTYPE)*N*N/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "=================================================\n\n";

    /********************************************************
     *  SYCL setup
     ********************************************************/
    sycl::gpu_selector device;
    sycl::queue queue(device);

    /********************************************************
     *  Data initilzation
     ********************************************************/
    // Input data
    std::vector<DTYPE> in(N*N);
    std::generate(in.begin(), in.end(), [](){return (std::rand()%10-5);});
    DTYPE* device_in = sycl::malloc_device<DTYPE>(N*N, queue);
    queue.memcpy(device_in, in.data(), N*N*sizeof(DTYPE));

    // Kernel data
    std::vector<DTYPE> kernel(KERNEL_SIZE*KERNEL_SIZE);
    std::generate(kernel.begin(), kernel.end(), [](){return (std::rand()%10-5);});
    DTYPE* device_kernel = sycl::malloc_device<DTYPE>(KERNEL_SIZE*KERNEL_SIZE, queue);
    queue.memcpy(device_kernel, kernel.data(), KERNEL_SIZE*KERNEL_SIZE*sizeof(DTYPE));


    // Output data
    std::vector<DTYPE> out(N*N);
    DTYPE* device_out = sycl::malloc_device<DTYPE>(N*N, queue);

    // For initial warming up
    queue.wait();

    /********************************************************
     *  Naive implementation
     ********************************************************/
    std::cout << "\nNaive parallel map operation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        stencil_naive<KERNEL_SIZE>(queue, device_in, device_kernel, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*N*N/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";
    std::cout << "-- Multiplications per second : "<<KERNEL_SIZE*KERNEL_SIZE*N*N/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" Gops\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, N*N*sizeof(DTYPE));
    queue.wait();
    check_result(in, kernel, out);
    #endif





    /********************************************************
     *  Finalize
     ********************************************************/
    sycl::free (device_in, queue);
    sycl::free (device_kernel, queue);
    sycl::free (device_out, queue);
    return 0;

    return 0;
}

void check_result(const std::vector<DTYPE>& in, const std::vector<DTYPE>& kernel, const std::vector<DTYPE>&out ) {
    
    int kernel_half_size = KERNEL_SIZE/2;
    DTYPE sum;
    int y, x, ky, kx;
    for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {

            // Get a gt value
            sum = 0;
            for (ky=-kernel_half_size; ky<=kernel_half_size; ky++) {
                for (kx=-kernel_half_size; kx<=kernel_half_size; kx++) {
                    if (0<=x+kx && x+kx<N && 0<=y+ky && y+ky<N) {
                        sum += in[(y+ky)*N+x+kx] * kernel[(ky+kernel_half_size)*KERNEL_SIZE+(kx+kernel_half_size)];
                    }
                }
            }

            // Check result
            if (out[y*N+x] != sum) {
                std::cout << "--- [[[ERROR]]] Checking the result failed at ["<<y<<","<<x<<"], gt("<<sum<<") != result("<<out[y*N+x]<<") !!\n";
                return ;
            }

        }
    }
            
    std::cout << "--- Checking the result succeed!!\n";
}