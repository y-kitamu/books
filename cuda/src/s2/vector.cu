#include <iostream>
#include <stdio.h>

constexpr int N = 5;

void cpuAdd(int *h_a, int *h_b, int *h_c) {
    int tid = 0;
    while (tid < N) {
        h_c[tid] = h_a[tid] + h_b[tid];
        tid += 1;
    }
}


__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
    int tid = blockIdx.x;
    if (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
        // printf("%d + %d = %d\n", d_a[tid], d_b[tid], d_c[tid]);
    }
}


int main() {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, sizeof(int) * N);
    cudaMalloc((void**)&d_b, sizeof(int) * N);
    cudaMalloc((void**)&d_c, sizeof(int) * N);

    printf("%p\n", d_a);
    printf("%p\n", d_b);
    printf("%p\n", d_c);

    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_h = clock();
    cpuAdd(h_a, h_b, h_c);
    clock_t end_h = clock();

    clock_t start_d = clock();
    gpuAdd<<<N, 1>>>(d_a, d_b, d_c);
    clock_t end_d = clock();

    double time_h = (double)(end_h - start_h) / CLOCKS_PER_SEC;
    double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;
    printf("No. of Elements in Array %d\n Device time %f seconds\n host time %f seconds\n",
           N, time_d, time_h);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    }

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
