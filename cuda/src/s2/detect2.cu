#include <stdio.h>
#include <cuda_runtime.h>


int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int device = 0; device < device_count; device++) {
        cudaDeviceProp device_property;
        cudaGetDeviceProperties(&device_property, device);

        memset(&device_property, 0, sizeof(device_property));
        device_property.major = 5;
        device_property.minor = 0;

        int dev;
        cudaChooseDevice(&dev, &device_property);
        printf("Id of device which version is 5.0 or greater : %d\n", dev);
    }
}
