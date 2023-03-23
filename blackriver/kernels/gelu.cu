#include "kernels.h"

namespace kernels {

__global__ void gelu(float* target, const float* src, int nElementNumber) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nElementNumber) {
        float value = src[i];
        target[i] = value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
    }
}

int gelu_forward(const float* src, float* target, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    gelu <<< num_of_blocks, block_size, 0, stream >>> (target, src, nElementNumber);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return 0;
}

}
