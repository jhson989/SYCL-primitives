#ifndef __JH_MAP_NAIVE__
#define __JH_MAP_NAIVE__


class MapFuncNaive {

    public:
        MapFuncNaive() : device_in(nullptr), device_out(nullptr) {}
        MapFuncNaive(DTYPE* d_in, DTYPE* d_out) : device_in(d_in), device_out(d_out) {}

        /*** Map operation  ***/


        /*** SYCL call interface ***/
        void operator() (sycl::nd_item<1> item) const {
            size_t x = item.get_global_id();
            device_out[x] = map(device_in[x]);
        }
        
        static const int OPS_PER_ITEM = 1;

    private:
        DTYPE* device_in;
        DTYPE* device_out;

};


void map_naive(sycl::queue& queue, DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(NUM_DATA, 1024), MapFuncNaive(device_in, device_out));
    });

    queue.wait();
}

#endif