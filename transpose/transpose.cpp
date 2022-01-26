#include <iostream>
#include <sys/time.h>
#include <vector>
#include <CL/sycl.hpp>
#include <algorithm>
namespace sycl=cl::sycl;

#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))
timeval start, end;

#define DTYPE float
const size_t M=10240, N=10240;
const size_t DIM_TILE=32;
const size_t WORK_PER_ITEM=1;
const size_t BLOCK_ROWS=DIM_TILE/WORK_PER_ITEM; // work per item is 4 (=32/8)


// Debugging info
#define __MODE_DEBUG_TIME__
const size_t NUM_TESTS=4;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&);

// Kernels
void naive_transpose(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out);


int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel Traspose\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 2D matrix transpose opertaion : in["<<M<<","<<N<<"] -> "<<"out["<<N<<","<<M<<"]\n";
    std::cout << "-- 2D matrix size: "<<sizeof(DTYPE)*M*N/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "=================================================\n\n";

    /********************************************************
     *  SYCL setup
     ********************************************************/
    sycl::gpu_selector device;
    sycl::queue queue(device);


    /********************************************************
     *  Data initilzation
     ********************************************************/
    std::vector<DTYPE> in(M*N);
    std::generate(in.begin(), in.end(), std::rand);
    DTYPE* device_in = sycl::malloc_device<DTYPE>(M*N, queue);
    queue.memcpy(device_in, in.data(), M*N*sizeof(DTYPE));
    queue.wait();

    std::vector<DTYPE> out(M*N);
    DTYPE* device_out = sycl::malloc_device<DTYPE>(M*N, queue);

    /********************************************************
     *  Naive implementation
     ********************************************************/
    std::cout << "Naive implementation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        naive_transpose(queue, device_in, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*M*N/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, M*N*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif


    /********************************************************
     *  Coalesced transpose via shared memory
     ********************************************************/
    std::cout << "Naive implementation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        naive_transpose(queue, device_in, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*M*N/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, M*N*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif


    /********************************************************
     *  No shared memory bank conflicts
     ********************************************************/
    std::cout << "Naive implementation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        //naive_transpose(queue, device_in, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*M*N/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, M*N*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif


    /********************************************************
     *  Finalize
     ********************************************************/
    return 0;
}

void naive_transpose(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(M*(N/WORK_PER_ITEM), DIM_TILE*BLOCK_ROWS), [=](sycl::nd_item<1> item){

            int y = item.get_global_id()/N*WORK_PER_ITEM;
            int x = item.get_global_id()%N;

            for (int work=0; work<WORK_PER_ITEM; work++) {
                device_out[x*M+(y+work)] = device_in[(y+work)*N+x];
            }
        });
    });


    queue.wait();
}


void check_result(const std::vector<DTYPE>& in, const std::vector<DTYPE>& out) {

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            if (in[y*N+x] != out[x*M+y]) {
                std::cout << "--- [[[ERROR]]] Checking the result failed!!\n";
                return ;
            }
        }
    }
    std::cout << "--- Checking the result succeed!!\n";
}