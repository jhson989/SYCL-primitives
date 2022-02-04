#ifndef __JH_MAP_WORK_INTENSIVE_UNROLLED__
#define __JH_MAP_WORK_INTENSIVE_UNROLLED__

class MapFuncWorkIntensiveUnrolled {

    public:
        MapFuncWorkIntensiveUnrolled() : device_in(nullptr), device_out(nullptr) {}
        MapFuncWorkIntensiveUnrolled(DTYPE* d_in, DTYPE* d_out) : device_in(d_in), device_out(d_out) {}

        /*** SYCL call interface ***/
        void operator() (sycl::nd_item<1> item) const {
            size_t x = item.get_global_id();
            size_t size = item.get_global_range()[0];

            device_out[x+0*size] = map(device_in[x+0*size]);
            device_out[x+1*size] = map(device_in[x+1*size]);
            device_out[x+2*size] = map(device_in[x+2*size]);
            device_out[x+3*size] = map(device_in[x+3*size]);
            device_out[x+4*size] = map(device_in[x+4*size]);
            device_out[x+5*size] = map(device_in[x+5*size]);
            device_out[x+6*size] = map(device_in[x+6*size]);
            device_out[x+7*size] = map(device_in[x+7*size]);
        }

    private:
        DTYPE* device_in;
        DTYPE* device_out;

};


void map_work_intensive_unrolled(sycl::queue& queue, DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(NUM_DATA/WORK_PER_ITEM, 1024), MapFuncWorkIntensiveUnrolled(device_in, device_out));
    });

    queue.wait();
}

#endif