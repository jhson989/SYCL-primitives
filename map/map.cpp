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
constexpr size_t NUM_DATA = 1<<27;

/*** Debugging info ***/
#define __MODE_DEBUG_TIME__
const size_t NUM_TESTS=20;
void check_result(const std::vector<DTYPE>&,const std::vector<DTYPE>&);

class MapFunc {

    public:
        MapFunc() : device_in(nullptr), device_out(nullptr) {}
        MapFunc(DTYPE* d_in, DTYPE* d_out) : device_in(d_in), device_out(d_out) {}

        /*** Map operation  ***/
        inline DTYPE map(const DTYPE in) const {
            return in+1;
        }

        /*** SYCL call interface ***/
        void operator() (sycl::nd_item<1> item) const {
            size_t x = item.get_global_id();
            device_out[x] = map(device_in[x]);
        }
        
    private:
        DTYPE* device_in;
        DTYPE* device_out;

};

void map_naive(sycl::queue& queue, DTYPE* device_in, DTYPE* device_out);



/********************************************************
 *  Main Function
 ********************************************************/
int main(void) {

    std::cout << "=================================================\n";
    std::cout << "SYCL Primitives : Parallel Map\n";
    std::cout << "-- a single nvidia GPU example\n";
    std::cout << "-- 1D vector map opertaion : in["<<NUM_DATA<<"] -> "<<"out["<<NUM_DATA<<"]\n";
    std::cout << "-- 1D vector size: "<<2*sizeof(DTYPE)*NUM_DATA/1024.0/1024.0/1024.0<<" GB\n";
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
    std::generate(in.begin(), in.end(), [](){return std::rand()%1000;});
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

    #ifdef __MODE_DEBUG_TIME__
    queue.memcpy(out.data(), device_out, NUM_DATA*sizeof(DTYPE));
    queue.wait();
    check_result(in, out);
    #endif


    return 0;
}

void map_naive(sycl::queue& queue, DTYPE* device_in, DTYPE* device_out) {

    queue.submit([&] (sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(NUM_DATA, 32), MapFunc(device_in, device_out));
    });

}

void check_result(const std::vector<DTYPE>& in, const std::vector<DTYPE>& out) {
    MapFunc MapFunctor;
    for (auto i=0; i!=in.size(); i++) {
        if (MapFunctor.map(in[i]) != out[i]) {
            std::cout << "--- [[[ERROR]]] Checking the result failed at ["<<i<<"] "<<MapFunctor.map(in[i])<<" != "<<out[i]<<" !!\n";
            return ;
        }
    }

    std::cout << "--- Checking the result succeed!!\n";
}