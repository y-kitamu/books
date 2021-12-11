#include <stdio.h>
#include <cuda_runtime.h>

constexpr int SIZE = 1000;
constexpr int NUM_BIN = 16;


__global__ void histogram_without_atomic(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        d_b[item]++;
    }
}


__global__ void histogram_atomic(int* d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];

    if (tid < SIZE) {
        atomicAdd(d_b + item, 1);
    }
}


__global__ void histogram_shared(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];

    __shared__ int cache[NUM_BIN];
    cache[threadIdx.x] = 0;
    __syncthreads();

    while (tid < SIZE) {
        atomicAdd(cache + item, 1);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();
    atomicAdd(d_b + threadIdx.x, cache[threadIdx.x]);
}


int main() {
    int h_a[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i % NUM_BIN;
    }

    int h_b[NUM_BIN];
    for (int i = 0; i < NUM_BIN; i++) {
        h_b[i] = 0;
    }

    int *d_a, *d_b;

    cudaMalloc((void**)&d_a, sizeof(int) * SIZE);
    cudaMalloc((void**)&d_b, sizeof(int) * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    histogram_atomic<<<(SIZE + NUM_BIN - 1) / NUM_BIN, NUM_BIN>>>(d_b, d_a);

    cudaMemcpy(h_b, d_b, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    printf("Histogram using 16 bin wihtout shared Memocy is : \n");

    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d : count : %d\n", i, h_b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}