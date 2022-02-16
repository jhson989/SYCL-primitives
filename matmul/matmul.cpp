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
const int M=1024*5+1, N=1024*5+11, K=1024*5+111;

/*** Debugging info ***/
#define __MODE_DEBUG_TIME__
const int NUM_TESTS=20;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&,const std::vector<DTYPE>&);

/*** Parallel algorithm implementations ***/
#include "includes/matmul_naive.hpp"
#include "includes/matmul_local_memory.hpp"


int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel 2D Matrix Multiplication\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 2D Matrix : A["<<M<<","<<K<<"] * B["<<K<<","<<N<<"] = C["<<M<<","<<N<<"]\n";
    std::cout << "-- total size of three 2D matrices: "<<sizeof(DTYPE)*(M*N+M*K+K*N)/1024.0/1024.0/1024.0<<" GB\n";
    std::cout << "=================================================\n\n";

    /********************************************************
     *  SYCL setup
     ********************************************************/
    sycl::gpu_selector device;
    sycl::queue queue(device);

    /********************************************************
     *  Data initilzation
     ********************************************************/
    // Input data A
    std::vector<DTYPE> A(M*K);
    std::generate(A.begin(), A.end(), [](){return (std::rand()%100-50);});
    DTYPE* device_A = sycl::malloc_device<DTYPE>(M*K, queue);
    queue.memcpy(device_A, A.data(), M*K*sizeof(DTYPE));

    // Input data B
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), [](){return (std::rand()%100-50);});
    DTYPE* device_B = sycl::malloc_device<DTYPE>(K*N, queue);
    queue.memcpy(device_B, B.data(), K*N*sizeof(DTYPE));
    
    // Output data C
    std::vector<DTYPE> C(M*N);
    DTYPE* device_C = sycl::malloc_device<DTYPE>(M*N, queue);

    // For initial warming up
    matmul_naive(queue, device_A, device_B, device_C, M, N, K);
    matmul_local_memory(queue, device_A, device_B, device_C, M, N, K);
    queue.wait();



    /********************************************************
     *  Naive implementation
     ********************************************************/
    std::cout << "\nNaive parallel matmul\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        matmul_naive(queue, device_A, device_B, device_C, M, N, K);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*(M*K+K*N+M*N)/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";
    std::cout << "-- Multiplications per second : "<<M/1024.0*N/1024.0*K/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" Gops\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(C.data(), device_C, M*N*sizeof(DTYPE));
    queue.wait();
    check_result(A, B, C);
    #endif


   /********************************************************
     *  Naive implementation
     ********************************************************/
    std::cout << "\nParallel matmul with local memory\n";
    gettimeofday(&start, NULL);
    for (int test=0; test<NUM_TESTS; test++){
        matmul_local_memory(queue, device_A, device_B, device_C, M, N, K);
    }   
    gettimeofday(&end, NULL);
    std::cout << "-- Elasped time : "<<ELAPSED_TIME(start, end)/NUM_TESTS<<" s\n";
    std::cout << "-- Effective bandwidth : "<<sizeof(DTYPE)*(M*K+K*N+M*N)/1024.0/1024.0/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" GB/s\n";
    std::cout << "-- Multiplications per second : "<<M/1024.0*N/1024.0*K/1024.0/(ELAPSED_TIME(start, end)/NUM_TESTS)<<" Gops\n";

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(C.data(), device_C, M*N*sizeof(DTYPE));
    queue.wait();
    check_result(A, B, C);
    #endif


    /********************************************************
     *  Finalize
     ********************************************************/
    sycl::free (device_A, queue);
    sycl::free (device_B, queue);
    sycl::free (device_C, queue);
    return 0;

    return 0;
}

void check_result(const std::vector<DTYPE>& A, const std::vector<DTYPE>& B, const std::vector<DTYPE>& C) {
    
    DTYPE sum;
    int m, n, k;
    for (m=0; m<M; m++) {
        for (n=0; n<N; n++) {
            
            sum = 0;
            for (k=0; k<K; k++)
                sum += A[m*K+k]*B[k*N+n];

            // Check result
            if (C[m*N+n] != sum) {
                std::cout << "--- [[[ERROR]]] Checking the result failed at ["<<m<<","<<n<<"], gt("<<sum<<") != result("<<C[m*N+n]<<") !!\n";
                return ;
            }

        }
    }
            
    std::cout << "--- Checking the result succeed!!\n";
}