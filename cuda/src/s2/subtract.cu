#include <iostream>
#include <stdio.h>


__global__ void subtract(int d_a, int d_b, int *d_c) {
    *d_c = d_a - d_b;
}


int main() {
    int h_a, h_b, h_c;
    int *d_c;

    h_a = 10;
    h_b = 7;

    cudaMalloc((void**)(&d_c), sizeof(int));

    subtract<<<1, 1>>>(h_a, h_b, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d - %d = %d\n", h_a, h_b, h_c);
    cudaFree(d_c);
}