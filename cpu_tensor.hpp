#ifndef _CPU_IMPL_HPP_
#define _CPU_IMPL_HPP_

#include "common.hpp"
#include "tensortype.hpp"

namespace br {

template <DataType _DTYPE_>
struct CPUTensor : public TransformerComputing {
    virtual ~CPUTensor() {
        if (mem_ != nullptr && owner_ ) {
            free(mem_);
        }
    }
    CPUTensor(const ShapeType& shape) : owner_(true) {
        if ( _DTYPE_ == DataType::Float ) {
            mem_ = malloc(shape.numel() * sizeof(float) );
        } else {
            br_panic("Can't be here!");
        }
    }
    CPUTensor(void *mem) : mem_(mem), owner_(false) { }

    void* data() {
        return mem_;
    }

    // Interfaces from TransformerComputing
    virtual ComputingReturn op_dump(tensor_t self) {
        size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
        if ( _DTYPE_ == DataType::Float ) {
            float* d = (float *)data();
            std::cout << "--------------------------" << std::endl;
            std::cout << "First " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;
            d = (float *)data() + self->items() - first8;
            std::cout << "Last " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;

            return OP_OK;
        }
        return OP_TODO_ERROR;
    }

    virtual ComputingReturn op_zero(tensor_t self) {
        if ( _DTYPE_ == DataType::Float ) {
            memset( mem_, 0, sizeof(float) * self->items() );
            return OP_OK;
        }
        return OP_TODO_ERROR;
    }

    virtual ComputingReturn op_fill(tensor_t self, float value) {
        if ( _DTYPE_ == DataType::Float ) {
            float* dst = (float *)data();
            for (size_t i = 0; i < self->items(); i++) {
                dst[i] = value;
            }
            return OP_OK;
        }
        return OP_TODO_ERROR;
    }

    virtual std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
        if ( _DTYPE_ == DataType::Float ) {
            ShapeType newShape(newShape_);
            float *newData = (float *)data() + offset;
            auto* newCpuTensor = new CPUTensor<DataType::Float>(newData);
            return std::make_shared<TensorType>(newCpuTensor, newShape);
        }

        return OP_TODO_ERROR;
    }
private:
    void* mem_;
    const bool owner_;

    friend struct CPUTensor<DataType::Float>;
    friend struct CPUTensor<DataType::BF16>;
};


} // end of namespace
#endif
