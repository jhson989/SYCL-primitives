#pragma once

template <typename T>
void matmul_naive(sycl::queue queue, const T* A, const T* B, T*C, const size_t M, const size_t N, const size_t K, const size_t gsize=16) {

    size_t ceil_M = ((M+gsize-1)/gsize)*gsize;
    size_t ceil_N = ((N+gsize-1)/gsize)*gsize;

    queue.submit([&] (sycl::handler& cgh) {

        cgh.parallel_for(sycl::nd_range<2>({ceil_M, ceil_N}, {gsize, gsize}), [=](sycl::nd_item<2> item) {

            int m = item.get_global_id(0);
            int n = item.get_global_id(1);
            if (m<M && n<N) {
                T sum = 0;
                for (int k=0; k<K; k++) {
                    sum += A[m*K+k]*B[k*N+n];
                }
                C[m*N+n] = sum;
            }
        });

    });

    queue.wait();

}