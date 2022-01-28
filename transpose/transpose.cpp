#include <iostream>
#include <sys/time.h>
#include <vector>
#include <CL/sycl.hpp>
#include <algorithm>
namespace sycl=cl::sycl;

#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))
timeval start, end;

#define DTYPE float
const size_t M=1024*30, N=1024*30;
const size_t DIM_TILE=32;
const size_t WORK_PER_ITEM=4;
const size_t BLOCK_ROWS=DIM_TILE/WORK_PER_ITEM;


// Debugging info
//#define __MODE_DEBUG_TIME__
const size_t NUM_TESTS=20;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&);

// Kernels
void transpose_naive(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out);
void transpose_shared_memory(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out);
void transpose_coalesced_shared_memory(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out);
void transpose_no_bank_conflict(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out);


int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel Traspose\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 2D matrix transpose opertaion : in["<<M<<","<<N<<"] -> "<<"out["<<N<<","<<M<<"]\n";
    std::cout << "-- 2D matrix size: "<<2*sizeof(DTYPE)*M*N/1024.0/1024.0/1024.0<<" GB\n";
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

    std::cout << "\nNaive implementation\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        transpose_naive(queue, device_in, device_out);
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
     *  Naive transpose via shared memory
     ********************************************************/

    std::cout << "\nNaive transpose via shared memory\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        transpose_shared_memory(queue, device_in, device_out);
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

    std::cout << "\nCoalesced transpose via shared memory\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        transpose_coalesced_shared_memory(queue, device_in, device_out);
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

    std::cout << "\nNo shared memory bank conflicts\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        transpose_no_bank_conflict(queue, device_in, device_out);
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

void transpose_naive(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out) {

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

void transpose_shared_memory(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        sycl::accessor<DTYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> local_in(sycl::range<2>(DIM_TILE,DIM_TILE), cgh);
        cgh.parallel_for(sycl::nd_range<2>({M,(N/WORK_PER_ITEM)}, {DIM_TILE,BLOCK_ROWS}), [=](sycl::nd_item<2> item){

            int y = item.get_global_id(1)*WORK_PER_ITEM;
            int x = item.get_global_id(0);
            int ly = item.get_local_id(1)*WORK_PER_ITEM;
            int lx = item.get_local_id(0);


            // Load input datain to local memory
            for (int work=0; work<WORK_PER_ITEM; work++) {
                local_in[ly+work][lx] = device_in[(y+work)*N+x];
            }

            // Synchronizing all the workitems in a group
            item.barrier(sycl::access::fence_space::local_space);

            for (int work=0; work<WORK_PER_ITEM; work++) {
                device_out[x*M+(y+work)] = local_in[ly+work][lx];
            }


        });
    });


    queue.wait();

}


void transpose_coalesced_shared_memory(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        sycl::accessor<DTYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> local_in(sycl::range<2>(DIM_TILE,DIM_TILE), cgh);
        cgh.parallel_for(sycl::nd_range<2>({M,(N/WORK_PER_ITEM)}, {DIM_TILE,BLOCK_ROWS}), [=](sycl::nd_item<2> item){

            int y = item.get_global_id(1)*WORK_PER_ITEM;
            int x = item.get_global_id(0);
            int ly = item.get_local_id(1)*WORK_PER_ITEM;
            int lx = item.get_local_id(0);


            // Load input datain to local memory
            for (int work=0; work<WORK_PER_ITEM; work++) {
                local_in[ly+work][lx] = device_in[(y+work)*N+x];
            }

            // Synchronizing all the workitems in a group
            item.barrier(sycl::access::fence_space::local_space);
            
            // Store input data into output
            int x_start = ((int)(x/DIM_TILE))*DIM_TILE;
            int y_start = ((int)(y/DIM_TILE))*DIM_TILE;
            x = y_start + lx;
            y = x_start + ly;

            for (int work=0; work<WORK_PER_ITEM; work++) {
                device_out[(y+work)*M+x] = local_in[lx][ly+work];
            }


        });
    });


    queue.wait();

}


void transpose_no_bank_conflict(sycl::queue& queue, const DTYPE* device_in, DTYPE* device_out) {


    queue.submit([&] (sycl::handler& cgh) {
        sycl::accessor<DTYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> local_in(sycl::range<2>(DIM_TILE+1,DIM_TILE), cgh);
        cgh.parallel_for(sycl::nd_range<2>({M,(N/WORK_PER_ITEM)}, {DIM_TILE,BLOCK_ROWS}), [=](sycl::nd_item<2> item){

            int y = item.get_global_id(1)*WORK_PER_ITEM;
            int x = item.get_global_id(0);
            int ly = item.get_local_id(1)*WORK_PER_ITEM;
            int lx = item.get_local_id(0);


            // Load input datain to local memory
            for (int work=0; work<WORK_PER_ITEM; work++) {
                local_in[ly+work][lx] = device_in[(y+work)*N+x];
            }

            // Synchronizing all the workitems in a group
            item.barrier(sycl::access::fence_space::local_space);
            
            // Store input data into output
            int x_start = ((int)(x/DIM_TILE))*DIM_TILE;
            int y_start = ((int)(y/DIM_TILE))*DIM_TILE;
            x = y_start + lx;
            y = x_start + ly;

            for (int work=0; work<WORK_PER_ITEM; work++) {
                device_out[(y+work)*M+x] = local_in[lx][ly+work];
            }


        });
    });



    queue.wait();

}



void check_result(const std::vector<DTYPE>& in, const std::vector<DTYPE>& out) {

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            if (in[y*N+x] != out[x*M+y]) {
                std::cout << "--- [[[ERROR]]] Checking the result failed at ["<<x<<","<<y<<"] "<<out[x*M+y]<<" !!\n";
                return ;
            }
        }
    }

    std::cout << "--- Checking the result succeed!!\n";
}