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
    Int = 3,
};

inline size_t DataType_size(DataType type_) {
    switch( type_ ) {
        case Int:
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
    const size_t operator[](int i) const {
        return dims_[i];
    }
    bool operator == (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return false;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other[i] != dims_[i] ) {
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
            if ( other[i] != dims_[i] ) {
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
using cpu_int_t = CPUTensor<DataType::Int>;
using cuda_float_t = CUDATensor<DataType::Float>;
using cuda_bf16_t = CUDATensor<DataType::BF16>;

// TensorType is all you need
struct TensorType: public TransformerComputing {
public:
    // init functions
    TensorType() = delete;
    TensorType(cpu_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), marker_(0), impl_(tensor) {};
    TensorType(cuda_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), marker_(0), impl_(tensor) {};
    TensorType(cpu_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), marker_(0), impl_(tensor) {};
    TensorType(cuda_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), marker_(0), impl_(tensor) {};
    TensorType(cpu_int_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Int), marker_(0), impl_(tensor) {};
    virtual ~TensorType();

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
    int64_t& marker() {
        return marker_;
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
    cpu_int_t* cpu_int() {
        if ( impl_.index() != CPU_INT ) {
            br_panic("Cant get cpu_int from a tensor");
        }
        return std::get<CPU_INT>(impl_);
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
        ss << device_name() << ":" <<  DataType_name( dtype() ) << ":" << marker_ ;
        ss << ":[";
        for (size_t i = 0; i < shape_.dim(); i++) {
            ss << shape_[i];
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
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_copy(tensor_t self, tensor_t src) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_transpos_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override ;
    ComputingReturn op_attn(tensor_t self, tensor_t v, tensor_t attn) override;
    ComputingReturn op_gelu(tensor_t self, tensor_t dst) override;
    ComputingReturn op_last_logits(tensor_t self, tensor_t mask,  tensor_t lm_head, tensor_t output) override;
    std::variant<ComputingReturn, float> op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) override;
    ComputingReturn op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) override;
    ComputingReturn op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) override;
    ComputingReturn op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) override;

    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_mpi_bcast(tensor_t self, int root) override;
    ComputingReturn io_mpi_recv(tensor_t self, int source) override;
    ComputingReturn io_nccl_recv(tensor_t self, int source) override;
    ComputingReturn io_nccl_send(tensor_t self, int dst) override;

private:
    // basic info about tensor
    ShapeType shape_;
    const DataType  dtype_;

    // help variable
    int64_t marker_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        CPU_FLOAT,
        CPU_BF16,
        CPU_INT,
        CUDA_FLOAT,
        CUDA_BF16,
    };
    using TensorImpl =   std::variant<  cpu_float_t*,
                                        cpu_bf16_t*,
                                        cpu_int_t*,
                                        cuda_float_t*,
                                        cuda_bf16_t* >;
    TensorImpl impl_;
};

tensor_t create_cuda_float(std::vector<size_t>& shape);
tensor_t create_cpu_float(std::vector<size_t>& shape);
tensor_t create_cpu_int(std::vector<size_t>& shape);

} // end of namespace br

#endif
