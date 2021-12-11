#include <stdio.h>


__global__ void multiply(int* d_a, int* d_b, int* d_c) {
    *d_c = (*d_a) * (*d_b);
}


int main() {
    int h_a, h_b, h_c;
    int *d_a, *d_b, *d_c;

    h_a = 10;
    h_b = 20;

    cudaMalloc((void**)(&d_a), sizeof(int));
    cudaMalloc((void**)(&d_b), sizeof(int));
    cudaMalloc((void**)(&d_c), sizeof(int));

    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(int), cudaMemcpyHostToDevice);

    multiply<<<500, 10>>>(d_a, d_b, d_c);
    multiply<<<10, 500>>>(d_a, d_b, d_c);
    multiply<<<50, 100>>>(d_a, d_b, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d * %d = %d\n", h_a, h_b, h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
