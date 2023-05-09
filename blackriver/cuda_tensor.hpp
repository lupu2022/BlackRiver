#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include "common.hpp"
#include "tensortype.hpp"

namespace br {

template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    virtual ~CUDATensor() {
        if (mem_ != nullptr && owner_) {
            CUDA_CHECK(cudaFree(mem_));
        }
    }

    CUDATensor(const ShapeType& shape) : owner_(true) {
        CUDA_CHECK(cudaMalloc(&mem_, shape.numel() * DataType_size(_DTYPE_)));
    }
    CUDATensor(void *mem) : mem_(mem), owner_(false) { }

    void* data() {
        return mem_;
    }

    cudnnTensorDescriptor_t create_cudnn_td_with(const std::vector<size_t> shape) {
        cudnnTensorFormat_t  format = CUDNN_TENSOR_NCHW;

        cudnnDataType_t dtype;
        cudnnTensorDescriptor_t desc;

        if ( _DTYPE_ == DataType::Float ) {
            dtype = CUDNN_DATA_FLOAT;
        } else {
            br_panic("cudnn don't support!");
        }

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

        if (shape.size() == 4) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], shape[2], shape[3]));
        } else if (shape.size() == 3) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, shape[2]));
        } else if (shape.size() == 2) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, 1));
        } else if (shape.size() == 1) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, 1, shape[0], 1, 1));
        } else {
            br_panic("cudnnSetTensor4dDescriptor: can't convert shape");
        }
        return desc;
    }

    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_nccl_send(tensor_t self, int dst) override;
    ComputingReturn io_nccl_recv(tensor_t self, int src) override;

    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_copy(tensor_t self, tensor_t src) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;

    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;

    ComputingReturn op_transpos_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override;
    ComputingReturn op_attn(tensor_t self, tensor_t value, tensor_t out) override;
    ComputingReturn op_gelu(tensor_t self, tensor_t dst) override;
    ComputingReturn op_last_logits(tensor_t self, tensor_t mask,  tensor_t lm_head, tensor_t output) override;

    std::variant<ComputingReturn, float> op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) override;
    ComputingReturn op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) override;
    ComputingReturn op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) override;
    ComputingReturn op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) override;
    ComputingReturn op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) override;
    ComputingReturn op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) override;


private:
    void*                       mem_;
    const bool                  owner_;

    friend struct CUDATensor<DataType::Float>;
    friend struct CUDATensor<DataType::BF16>;
};


} // end of namespace
#endif
