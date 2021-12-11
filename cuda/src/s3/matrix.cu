#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

constexpr int TILE_SIZE = 2;

__global__ void matmul_no_shared(float *d_a, float *d_b, float *d_c, const int size) {
    int row, col;
    row = TILE_SIZE * blockIdx.x + threadIdx.x;
    col = TILE_SIZE * blockIdx.y + threadIdx.y;

    for (int k = 0; k < size; k++) {
        d_c[row * size + col] += d_a[row * size + k] * d_b[k * size + col];
    }
}

__global__ void matmul_shared(float *d_a, float *d_b, float *d_c, const int size) {
    int row, col;
    row = TILE_SIZE * blockIdx.y + threadIdx.y;
    col = TILE_SIZE * blockIdx.x + threadIdx.x;

    __shared__ float shared_a[TILE_SIZE][TILE_SIZE], shared_b[TILE_SIZE][TILE_SIZE];

    for (int i = 0; i < size / TILE_SIZE; i++) {
        shared_a[threadIdx.y][threadIdx.x] = d_a[row * size + (i * TILE_SIZE + threadIdx.x)];
        shared_b[threadIdx.y][threadIdx.x] = d_b[(i * TILE_SIZE + threadIdx.y) * size + col];
        __syncthreads();

        if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("%d, %d, %f\n", threadIdx.y, threadIdx.x, shared_a[threadIdx.y][threadIdx.x]);
        }

        for (int j = 0; j < TILE_SIZE; j++) {
            d_c[row * size + col] += shared_a[threadIdx.x][j] * shared_b[j][threadIdx.y];
        }
        __syncthreads();
    }
}


int main() {
    const int size = 6;

    float h_a[size][size], h_b[size][size], h_result[size][size];
    float *d_a, *d_b, *d_result;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_a[i][j] = j;
            h_b[i][j] = j;
        }
    }

    cudaMalloc((void**)&d_a, size * size * sizeof(float));
    cudaMalloc((void**)&d_b, size * size * sizeof(float));
    cudaMalloc((void**)&d_result, size * size * sizeof(float));

    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(size / TILE_SIZE, size / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // matmul_no_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);
    matmul_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_result, size);

    cudaMemcpy(h_result, d_result, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("The result of Matrix multiplication is :\n");

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", h_result[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}