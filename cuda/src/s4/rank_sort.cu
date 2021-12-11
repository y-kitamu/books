#include <device_launch_parameters.h>
#include <stdio.h>


constexpr int arraySize = 5;
constexpr int threadPerBlock = 5;


__global__ void addKernel(int *d_a, int *d_b) {
    int count = 0;
    int tid = threadIdx.x;
    int ttid = blockIdx.x * threadPerBlock + tid;
    int val = d_a[ttid];

    __shared__ int cache[threadPerBlock];
    for (int i = tid; i < arraySize; i += threadPerBlock) {
        cache[tid] = d_a[i];
        __syncthreads();
        for (int j = 0; j < threadPerBlock; ++j) {
            if (val > cache[j]) {
                count++;
            }
        }
        __syncthreads();
    }
    d_b[count] = val;
}


int main() {
    int h_a[arraySize] = {5, 9, 3, 4, 8};
    int h_b[arraySize];
    int *d_a, *d_b;

    cudaMalloc((void**)&d_a, sizeof(int) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(int) * arraySize);

    cudaMemcpy(d_a, h_a, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

    addKernel<<<arraySize / threadPerBlock, threadPerBlock>>>(d_a, d_b);

    cudaMemcpy(h_b, d_b, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    printf("The Enumeration sorted Array is: \n");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}