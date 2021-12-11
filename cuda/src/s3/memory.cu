#include <stdio.h>


constexpr int N = 5;


__global__ void gpu_global_memory(int *d_a) {
    d_a[threadIdx.x] = threadIdx.x;
}


int main(int argc, char ** argv) {
    int h_a[N];
    int *d_a;

    cudaMalloc((void**)&d_a, sizeof(int) * N);

    gpu_global_memory<<<1, N>>>(d_a);

    cudaMemcpy(h_a, d_a, sizeof(int) * N, cudaMemcpyDeviceToHost);

    printf("Array in Global memory is: \n");

    for (int i = 0; i < N; i++) {
        printf("At Index : %d --> %d\n", i, h_a[i]);
    }

    cudaFree(d_a);
}