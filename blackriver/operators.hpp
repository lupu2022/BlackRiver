#ifndef _OPERATORS_HPP_
#define _OPERATORS_HPP_

#include <vector>
#include <memory>

namespace br {
struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

// low level API for implementing Transformer
enum ComputingReturn {
    OP_OK = 0,
    OP_TODO_ERROR = -1,
    OP_INPUT_ERROR = -2,
    OP_OUTPUT_ERROR = -3,
    OP_ATTR_ERROR = -4,
};

struct TransformerComputing {
    virtual ComputingReturn io_load(tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_save(tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_dump(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_bcast(tensor_t self, int root) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_recv(tensor_t self, int source) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_send(tensor_t self, int dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_nccl_recv(tensor_t self, int source) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_nccl_send(tensor_t self, int dst) {
        return OP_TODO_ERROR;
    }

    virtual ComputingReturn op_zero(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_fill(tensor_t self, float value) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_copy(tensor_t self, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t bias, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_transpos_0213(tensor_t self, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_softmax(tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_attn(tensor_t self, tensor_t v, tensor_t attn) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_gelu(tensor_t self, tensor_t dst) {
        return OP_TODO_ERROR;
    }
};


} // endof namepsapce tt
#endif
