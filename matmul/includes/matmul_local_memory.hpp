#pragma once

template <typename T>
void matmul_local_memory(sycl::queue queue, const T* A, const T* B, T*C, const size_t M, const size_t N, const size_t K, const size_t gsize=16) {

    size_t ceil_M = ((M+gsize-1)/gsize)*gsize;
    size_t ceil_N = ((N+gsize-1)/gsize)*gsize;
    size_t ceil_K = ((K+gsize-1)/gsize)*gsize;

    queue.submit([&] (sycl::handler& cgh) {

        
        sycl::accessor<T, 2, sycl::access::mode::read_write, sycl::access::target::local> local_A(sycl::range<2>(gsize, gsize), cgh);
        sycl::accessor<T, 2, sycl::access::mode::read_write, sycl::access::target::local> local_B(sycl::range<2>(gsize, gsize), cgh);
        cgh.parallel_for(sycl::nd_range<2>({ceil_M, ceil_N}, {gsize, gsize}), [=](sycl::nd_item<2> item) {

            int m = item.get_global_id(0);
            int n = item.get_global_id(1);

            int lm = item.get_local_id(0);
            int ln = item.get_local_id(1);
            
            int k;
            T sum = 0;
            for (int tile=0; tile<ceil_K; tile+=gsize) {

                if (m<M && ln+tile<K)
                    local_A[lm][ln] = A[m*K+(ln+tile)];

                if (lm+tile<K && n<N)
                    local_B[lm][ln] = B[(lm+tile)*N+n];
                item.barrier(sycl::access::fence_space::local_space);


                for (k=0; k<gsize && k+tile<K; k++) {
                    sum += local_A[lm][k]*local_B[k][ln];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (m<M && n<N)
                C[m*N+n] = sum;
        });

    });

    queue.wait();

}
