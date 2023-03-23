#ifndef _TENSORTYPE_HPP_
#define _TENSORTYPE_HPP_

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include <variant>

#include "common.hpp"
#include "operators.hpp"

namespace br {

enum DataType {
    Float = 0,
    BF16 = 1,
    F16 = 2,
};

inline size_t DataType_size(DataType type_) {
    switch( type_ ) {
        case Float:
            return 4;
        case F16:
        case BF16:
            return 2;
        default:
            break;
    }
    br_panic("Can't be here");
    return 0;
}

inline const char* DataType_name(DataType type_) {
    switch( type_ ) {
        case Float:
            return "f32";
        case F16:
            return "f16";
        case BF16:
            return "bf16";
        default:
            break;
    }
    br_panic("Can't be here");
    return NULL;
}

// Logical/Math shape of a tensor
struct ShapeType {
public:
    ShapeType() = delete;
    ShapeType(const std::vector<size_t>& dims) {
        size_t ND = dims.size();
        dims_.resize(ND);
        for(size_t i = 0; i < ND; i++) {
            dims_[i] = dims[i];
        }
        numel_ = 0;
    }
    // all kinds accessors
    size_t numel() const {
        if ( numel_ != 0) {
            return numel_;
        }
        numel_ = 1;
        for(size_t i = 0; i < dims_.size(); i++) {
            numel_ *= dims_[i];
        }
        return numel_;
    }
    const std::vector<size_t>& vec() const {
        return dims_;
    }
    const size_t* dims() const {
        return &dims_[0];
    }
    const size_t dim() const {
        return dims_.size();
    }
    bool operator == (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return false;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other.vec()[i] != dims_[i] ) {
                return false;
            }
        }
        return true;
    }
    bool operator != (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return true;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other.vec()[i] != dims_[i] ) {
                return true;
            }
        }
        return false;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < dim(); i++) {
            ss << dims_[i] << " ";
        }
        ss << "]";
        return ss.str();
    }

private:
    std::vector<size_t>  dims_;
    mutable size_t numel_;
};

// forward declare
template <DataType _DTYPE_> struct CPUTensor;
template <DataType _DTYPE_> struct CUDATensor;
using cpu_float_t = CPUTensor<DataType::Float>;
using cpu_bf16_t = CPUTensor<DataType::BF16>;
using cuda_float_t = CUDATensor<DataType::Float>;
using cuda_bf16_t = CUDATensor<DataType::BF16>;

// TensorType is all you need
struct TensorType: public TransformerComputing {
public:
    // init functions
    TensorType() = delete;
    TensorType(cpu_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(cuda_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(cpu_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), impl_(tensor) {};
    TensorType(cuda_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), impl_(tensor) {};
    ~TensorType();

    // fast access
    const ShapeType& shape() const {
        return shape_;
    }
    const DataType& dtype() const {
        return dtype_;
    }
    const size_t items() {
        return shape_.numel();
    }
    size_t impl_index() const {
        return impl_.index();
    }
    cpu_float_t* cpu_float() {
        if ( impl_.index() != CPU_FLOAT ) {
            br_panic("Cant get cpu_float from a tensor");
        }
        return std::get<CPU_FLOAT>(impl_);
    }
    cpu_bf16_t* cpu_bf16() {
        if ( impl_.index() != CPU_BF16 ) {
            br_panic("Cant get cpu_bf16 from a tensor");
        }
        return std::get<CPU_BF16>(impl_);
    }
    cuda_float_t* cuda_float() {
        if ( impl_.index() != CUDA_FLOAT ) {
            br_panic("Cant get cuda_float from a tensor");
        }
        return std::get<CUDA_FLOAT>(impl_);
    }
    cuda_bf16_t* cuda_bf16() {
        if ( impl_.index() != CUDA_BF16 ) {
            br_panic("Cant get cuda_bf16 from a tensor");
        }
        return std::get<CUDA_BF16>(impl_);
    }

    // help functions
    std::string to_string() {
        std::stringstream ss;
        ss << device_name() << ":" <<  DataType_name( dtype() );
        ss << ":[";
        for (size_t i = 0; i < shape_.vec().size(); i++) {
            ss << shape_.vec()[i];
            if (i != shape_.dim() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }
    const char* device_name() {
        if (impl_index() == ImplType::CPU_FLOAT) {
            return "cpu";
        }
        if (impl_index() == ImplType::CPU_BF16) {
            return "cpu";
        }
        return "cuda";
    }

    bool is_cpu() const {
        if (impl_index() == ImplType::CPU_FLOAT) {
            return true;
        }
        if (impl_index() == ImplType::CPU_BF16) {
            return true;
        }
        return false;
    }

    bool is_cuda() const {
        if (impl_index() == ImplType::CUDA_FLOAT) {
            return true;
        }
        if (impl_index() == ImplType::CUDA_BF16) {
            return true;
        }
        return false;
    }

    bool same_impl(tensor_t& other) {
        if ( impl_index() != other->impl_index() ) {
            return false;
        }
        return true;
    }
    bool same_dtype(tensor_t& other) {
        if ( dtype_ != other->dtype() ) {
            return false;
        }
        return true;
    }
    bool same_shape(tensor_t& other) {
        if ( shape_ != other->shape() ) {
            return false;
        }
        return true;
    }

    TransformerComputing* impl();

public:
#include "tensortype-impl.hpp"

private:
    // basic info about tensor
    ShapeType shape_;
    const DataType  dtype_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        CPU_FLOAT,
        CPU_BF16,
        CUDA_FLOAT,
        CUDA_BF16,
    };
    using TensorImpl =   std::variant<  cpu_float_t*,
                                        cpu_bf16_t*,
                                        cuda_float_t*,
                                        cuda_bf16_t* >;

    TensorImpl impl_;
};


} // end of namespace br

#endif
