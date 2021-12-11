#include <memory>
#include <iostream>
#include <cuda_runtime.h>


int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Ther are no available device(s) that support Cuda\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", device_count);
    }

    for (int device = 0; device < device_count; device++) {
        cudaDeviceProp device_property;
        cudaGetDeviceProperties(&device_property, device);
        // global properties
        printf("\nDevice %d: \"%s\"\n", device, device_property.name);
        printf("  version = %d.%d\n", device_property.major, device_property.minor);
        int driver_version, runtime_version;
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        printf("  Duca Driver Version / Runtime Version %d.%d / %d.%d\n",
               driver_version / 1000, (driver_version % 100) / 10,
               runtime_version / 1000, (runtime_version % 100) / 10);
        printf("  Total amount of global memory : %.0f Bytes (%llu bytes)\n",
               (float)device_property.totalGlobalMem / 1048576.0f,
               (unsigned long long)device_property.totalGlobalMem);
        printf("  (%2d) Multiprocessors", device_property.multiProcessorCount);
        printf("  GPU Max Clock rate : %.0fMHz (%0.2f GHz)\n",
               device_property.clockRate * 1e-3f, device_property.clockRate * 1e-6f);

        // memory related properties
        printf("  Total amount of global memory : %.0f MBytes (%llu bytes)\n",
               (float)device_property.totalGlobalMem / 1048576.0f,
               (unsigned long long)device_property.totalGlobalMem);
        printf("  Memory Clock rate: %.0f Mhz\n", device_property.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width: %d-bit\n", device_property.memoryBusWidth);
        if (device_property.l2CacheSize) {
            printf("  L2 Cache Size : %d bytes\n", device_property.l2CacheSize);
        }
        printf("  Total amount of constant memory : %lu bytes", device_property.totalConstMem);
        printf("  Total amount of shared memory per block: %lu bytes\n",
               device_property.sharedMemPerBlock);
        printf("  Total amount of registers available per block: %d\n", device_property.regsPerBlock);

        // thread related properties
        printf("  Maximum number of threads per multiprocessor : %d\n",
               device_property.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block : %d\n",
               device_property.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x, y, z) : (%d, %d, %d)\n",
               device_property.maxThreadsDim[0],
               device_property.maxThreadsDim[1],
               device_property.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size (x, y, z) : (%d, %d, %d)\n",
               device_property.maxGridSize[0],
               device_property.maxGridSize[1],
               device_property.maxGridSize[2]);
    }

}