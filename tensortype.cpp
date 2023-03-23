#include "common.hpp"
#include "tensortype.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

namespace br {

TensorType::~TensorType() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_BF16 ) {
        cuda_bf16_t* tensor = std::get<CUDA_BF16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CPU_FLOAT ) {
        cpu_float_t* tensor = std::get<CPU_FLOAT>(impl_);
        delete tensor;
    }
}

TransformerComputing* TensorType::impl() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_BF16 ) {
        cuda_bf16_t* tensor = std::get<CUDA_BF16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CPU_FLOAT ) {
        cpu_float_t* tensor = std::get<CPU_FLOAT>(impl_);
        return tensor;
    }

    br_panic("Can't be here!");
    return nullptr;
}

}

