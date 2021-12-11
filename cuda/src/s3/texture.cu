#include <stdio.h>

constexpr int NUM_THREADS = 10;
constexpr int N = 10;

texture<float, 1, cudaReadModeElementType> textureRef;

__global__ void gpu_texture_memory(int  n, float * d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = tex1D(textureRef, float(idx));
        d_out[idx] = temp;
    }
}


int main() {
    int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);

    float *d_out;

    cudaMalloc((void**)&d_out, sizeof(float) * N);

    float *h_out = (float*)malloc(sizeof(float) * N);
    float h_in[N];

    for (int i = 0; i < N; i++) {
        h_in[i] = float(i);
    }

    cudaArray *cu_array;
    cudaMallocArray(&cu_array, &textureRef.channelDesc, N, 1);

    cudaMemcpyToArray(cu_array, 0, 0, h_in, sizeof(float)*N, cudaMemcpyHostToDevice);

    cudaBindTextureToArray(textureRef, cu_array);

    gpu_texture_memory<<<num_blocks, NUM_THREADS>>>(N, d_out);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("Use of Texture memory on GPU\n");

    for (int i = 0; i < N; i++) {
        printf("Average between two nearest element is %f\n", h_out[i]);
    }

    free(h_out);
    cudaFree(d_out);
    cudaFreeArray(cu_array);
    cudaUnbindTexture(textureRef);
}
