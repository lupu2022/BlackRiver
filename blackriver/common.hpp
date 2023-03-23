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
#include <msgpack.hpp>

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
void load_data(const char* weights_file, std::vector<T> &allPerform) {
    std::ifstream t(weights_file, std::ios::binary);

    t.seekg(0, std::ios::end);
    size_t totalSize = t.tellg();
    t.seekg(0, std::ios::beg);

    char* memblock;
    memblock = new char [totalSize];
    t.read(memblock, totalSize);
    t.close();

    try {
        auto oh = msgpack::unpack((const char*)memblock, totalSize);
        allPerform = oh.get().as<std::vector<T>>();
        delete memblock;
    }
    catch (...) {
        std::cout << "Unpack weight error!" << std::endl;
        assert(false);
    }
}


}
#endif

