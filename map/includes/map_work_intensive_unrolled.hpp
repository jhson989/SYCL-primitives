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
            size_t now = size;
            

            device_out[x] = map(device_in[x]);
            device_out[x+now] = map(device_in[x+now]); now += size;
            device_out[x+now] = map(device_in[x+now]); now += size;
            device_out[x+now] = map(device_in[x+now]); now += size;

            device_out[x+now] = map(device_in[x+now]); now += size;
            device_out[x+now] = map(device_in[x+now]); now += size;
            device_out[x+now] = map(device_in[x+now]); now += size;
            device_out[x+now] = map(device_in[x+now]);
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