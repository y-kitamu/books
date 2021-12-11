#include <stdio.h>

constexpr int NUM_THREADS = 100000;
constexpr int SIZE = 10;

constexpr int BLOCK_WIDTH = 100;


__global__ void gpu_increment_without_atomic(int* d_a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    tid = tid % SIZE;
    d_a[tid] += 1;
}


__global__ void gpu_increment_atomic(int *d_a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    tid = tid % SIZE;
    atomicAdd(&d_a[tid], 1);
}


int main() {
    printf("%d total threads in %d blocks writiing into %d array elements\n",
           NUM_THREADS, BLOCK_WIDTH, SIZE);

    int h_a[SIZE];
    int *d_a;

    const int ARRAY_BYTES = SIZE * sizeof(int);
    cudaMalloc((void**)&d_a, ARRAY_BYTES);
    cudaMemset((void*)d_a, 0, ARRAY_BYTES);

    // gpu_increment_without_atomic<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);
    gpu_increment_atomic<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);

    cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    printf("Number of times a particular Array index has been incremented without atomic add is :\n");
    for (int i = 0; i < SIZE;i ++) {
        printf("index: %d --> %d times\n", i, h_a[i]);
    }
    cudaFree(d_a);
    return 0;
}
