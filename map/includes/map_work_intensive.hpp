#ifndef __JH_MAP_WORK_INTENSIVE__
#define __JH_MAP_WORK_INTENSIVE__


class MapFuncWorkIntensive {

    public:
        MapFuncWorkIntensive() : device_in(nullptr), device_out(nullptr) {}
        MapFuncWorkIntensive(DTYPE* d_in, DTYPE* d_out, int wpi) : device_in(d_in), device_out(d_out), WORK_PER_ITEM(wpi) {}

        /*** Map operation  ***/


        /*** SYCL call interface ***/
        void operator() (sycl::nd_item<1> item) const {
            size_t x = item.get_global_id();
            size_t size = item.get_global_range()[0];
            for (int i=0; i<WORK_PER_ITEM; i++)
                device_out[x+i*size] = map(device_in[x+i*size]);
        }
        
        static const int OPS_PER_ITEM = 1;

    private:
        DTYPE* device_in;
        DTYPE* device_out;
        int WORK_PER_ITEM;

};


void map_work_intensive(sycl::queue& queue, DTYPE* device_in, DTYPE* device_out, int WORK_PER_ITEM) {

    queue.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(NUM_DATA/WORK_PER_ITEM, 1024), MapFuncWorkIntensive(device_in, device_out, WORK_PER_ITEM));
    });

    queue.wait();
}

#endif