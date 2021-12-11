#include <iostream>

__global__ void myfirstkernel() {}


int main() {
    myfirstkernel<<<1, 1>>>();
    printf("Hello, Cuda\n");
    return 0;
}
