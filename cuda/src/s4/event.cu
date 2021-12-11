#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int N = 100000;


__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += gridDim.x * blockDim.x;
    }
}


int main() {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaEventRecord(e_start, 0);

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    gpuAdd<<<512, 512>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("Time to add %d numbers : %3.1f ms\n", N, elapsedTime);
}
