#include <stdio.h>

constexpr int N = 50;

__global__ void cube(int* d_in, int* d_out) {
    int tid = threadIdx.x;
    if (tid < N) {
        int tmp = d_in[tid];
        d_out[tid] = tmp * tmp * tmp;
    }
}


int main() {
    int h_in[N], h_out[N];
    int *d_in, *d_out;

    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    cudaMalloc((void**)(&d_in), sizeof(int) * 50);
    cudaMalloc((void**)(&d_out), sizeof(int) * 50);

    cudaMemcpy(d_in, &h_in, sizeof(int) * N, cudaMemcpyHostToDevice);

    cube<<<1, N>>>(d_in, d_out);

    cudaMemcpy(&h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("cube of %d is %d\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}
