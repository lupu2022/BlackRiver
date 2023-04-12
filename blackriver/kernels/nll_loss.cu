#include <cooperative_groups.h>
#include <cstddef>

#include "common.h"
#include "block_reduce.h"
#include "kernels.h"

namespace kernels {

__global__ void nll_loss(const int* ids, const float* logsoftmax, float *output, int n, int vocab ) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = 0.0;
    if ( i < n ) {
        int index = ids[i];
        value = logsoftmax[ i * vocab +  index]; 
    }

    float reduce_val[1] = {value};
    blockReduce<ReduceType::kSum, 1>(reduce_val);
    
    if ( i == 0 ) {
        *output = reduce_val[0];
    }
}

int nllloss_forward(const int* ids, const float* logsoftmax, float *output, int n, int vocab, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    nll_loss <<< num_of_blocks, block_size, 0, stream >>> (ids, logsoftmax, output, n, vocab);
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
 
    return 0;
}

}
