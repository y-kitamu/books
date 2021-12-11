#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


constexpr int N = 5;


__global__ void gpuSquare(float *d_in, float *d_out) {
    int tid = threadIdx.x;
    if (tid < N) {
        // d_out = d_in[tid] * d_in[tid];
        float temp = d_in[tid];
        d_out[tid] = temp * temp;
    }
}


int main() {
    float h_in[N], h_out[N];
    float *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);

    gpuSquare<<<1, N>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("The square of %f is %f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}
