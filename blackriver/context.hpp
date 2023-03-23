#ifndef _CONTEXT_HPP_
#define _CONTEXT_HPP_

#include "common.hpp"

namespace br {

struct ComputingContext {
    static int cuda_device;
    static cudaStream_t cuda_stream;
    static cublasHandle_t cublas_handle;
    static cublasLtHandle_t cublasLt_handle;
    static cudnnHandle_t cudnn_handle;

    static void* cuda_workspace;
    static size_t cuda_workspace_size;

    static void boot(int cud);
    static void shutdown();
};


} // end of namespace br


#endif
