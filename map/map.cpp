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
#define DTYPE float
constexpr size_t NUM_DATA = 1<<29;
#define WORK_PER_ITEM 32

/*** Debugging info ***/
#define __MODE_DEBUG_TIME__
const size_t NUM_TESTS=20;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&);

inline DTYPE map(const DTYPE in) {
    return in-1;
}

/*** Map inplementation ***/
#include "includes/map_naive.hpp"
#include "includes/map_work_intensive.hpp"
#include "includes/map_work_intensive_unrolled.hpp"

/********************************************************
 *  Main Function
 ********************************************************/
int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel Map\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 1D vector map opertaion : in["<<NUM_DATA<<"] -> "<<"out["<<NUM_DATA<<"]\n";
    std::cout << "-- 1D vector size: "<<sizeof(DTYPE)*NUM_DATA/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "-- test environment : NVIDIA RTX 2060 super (bandwidth: 448.0 GB/s)\n";
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
    std::vector<DTYPE> in(NUM_DATA);
    std::generate(in.begin(), in.end(), std::rand);
    DTYPE* device_in = sycl::malloc_device<DTYPE>(NUM_DATA, queue);
    queue.memcpy(device_in, in.data(), NUM_DATA*sizeof(DTYPE));
    queue.wait();

    // Output data
    std::vector<DTYPE> out(NUM_DATA);
    DTYPE* device_out = sycl::malloc_device<DTYPE>(NUM_DATA, queue);


    /********************************************************
     *  Naive implementation
     ********************************************************/
    std::cout << "\nNaive parallel map operation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        map_naive(queue, device_in, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*NUM_DATA/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";
    std::cout << "-- Operations per second : "<<MapFuncNaive::OPS_PER_ITEM*NUM_DATA/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" Gops\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, NUM_DATA*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif

    /********************************************************
     *  Work intensive implementation
     ********************************************************/
    std::cout << "\nWork intensive parallel map operation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        map_work_intensive(queue, device_in, device_out);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*NUM_DATA/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";
    std::cout << "-- Operations per second : "<<MapFuncNaive::OPS_PER_ITEM*NUM_DATA/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" Gops\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, NUM_DATA*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif



    /********************************************************
     *  Finalize
     ********************************************************/
    sycl::free (device_in, queue);
    sycl::free (device_out, queue);
    return 0;
}

void check_result(const std::vector<DTYPE>& in, const std::vector<DTYPE>& out) {
    for (auto i=0; i!=in.size(); i++) {
        if (map(in[i]) != out[i]) {
            std::cout << "--- [[[ERROR]]] Checking the result failed at ["<<i<<"] "<<map(in[i])<<" != "<<out[i]<<" !!\n";
            return ;
        }
    }

    std::cout << "--- Checking the result succeed!!\n";
}