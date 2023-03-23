#include "common.hpp"
#include "computing.hpp"

namespace br {

int ComputingContext::cuda_device = -1;
cudaStream_t ComputingContext::cuda_stream = nullptr;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
cudnnHandle_t ComputingContext::cudnn_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
size_t ComputingContext::cuda_workspace_size = 0;

void ComputingContext::boot(int cud) {
    cuda_device = cud;

    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cuda_stream) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));

    cuda_workspace_size = 1024 * 1024 * 32;
    CUDA_CHECK( cudaMalloc(&cuda_workspace, cuda_workspace_size) );
}

void ComputingContext::shutdown() {
    CUDNN_CHECK( cudnnDestroy(cudnn_handle) );
    CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
}

}
