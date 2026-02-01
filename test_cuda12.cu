#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel() {
    printf("Hello from CUDA 12.6 on Pascal GPU!\n");
}

int main() {
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
