#include <iostream>
#include <stdio.h>


__global__ void myfirstkernel(void) {
    printf("Hello! I'm thread id : %d, in block : %d\n", threadIdx.x, blockIdx.x);
}


int main() {
    myfirstkernel<<<16, 1>>>();
    cudaDeviceSynchronize();
    printf("Finish!\n");
    return 0;
}
