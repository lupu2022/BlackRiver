#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <vector>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#define COMPLAIN_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns error: %d.\n", __FILE__, __LINE__, \
                what, status); \
        exit(1); \
    } while (0)

#define CUDA_CHECK(f) \
    do { \
        cudaError_t  s_ = f; \
        if (s_ != cudaSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CUBLAS_CHECK(f) \
    do { \
        cublasStatus_t  s_ = f; \
        if (s_ != CUBLAS_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CUDNN_CHECK(f) \
    do { \
        cudnnStatus_t  s_ = f; \
        if (s_ != CUDNN_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define MPI_CHECK(f) \
    do { \
        int  s_ = f; \
        if (s_ != MPI_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define NCCL_CHECK(f) \
    do { \
        ncclResult_t  s_ = f; \
        if (s_ != ncclSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)


#define br_assert(Expr, Msg) \
    br::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define br_panic(Msg) \
    br::_M_Panic(__FILE__, __LINE__, Msg)

#define op_check(ret, Msg)                     \
    if ( ret != br::OP_OK ) {                  \
        br::_M_Panic(__FILE__, __LINE__, Msg); \
    }                                          \
    return ret

namespace br {

// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "**Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "**Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}

template<typename T>
inline void read_data(const char* fileName, std::vector<T>& dout) {
    std::ifstream inf(fileName, std::ios::binary);
    inf.seekg(0, inf.end);
    size_t length = inf.tellg();
    inf.seekg(0, inf.beg);

    const size_t count = 1024;
    const size_t items = length / sizeof(T);
    br_assert( items % count == 0, "Only support block read");
    dout.resize( items );

    for(size_t i = 0; i < items / count; i++) {
        T* src = (T *)dout.data() + i * count;
        inf.read((char *)src , sizeof(T) * count);
    }

    inf.close();
}

}

#endif

