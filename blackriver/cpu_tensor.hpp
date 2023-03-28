#ifndef _CPU_IMPL_HPP_
#define _CPU_IMPL_HPP_

#include "context.hpp"
#include "common.hpp"
#include "tensortype.hpp"

namespace br {

template <DataType _DTYPE_>
struct CPUTensor : public TransformerComputing {
    virtual ~CPUTensor() { }
    CPUTensor(const ShapeType& shape) {
        if ( _DTYPE_ == DataType::Float ) {
            mem_ = MemoryContext::alloc(shape.numel() * sizeof(float));
        } else {
            br_panic("Can't be here!");
        }
    }
    CPUTensor(void *mem) : mem_(mem){ }

    void* data() {
        return mem_;
    }

public:
    // Interfaces from TransformerComputing
    virtual ComputingReturn io_dump(tensor_t self);
    virtual ComputingReturn io_load(tensor_t self, const char* fileName);
    virtual ComputingReturn io_save(tensor_t self, const char* fileName);
    virtual ComputingReturn io_mpi_recv(tensor_t self, int source);

    virtual ComputingReturn op_zero(tensor_t self);
    virtual ComputingReturn op_copy(tensor_t self, tensor_t dst);
    virtual ComputingReturn op_fill(tensor_t self, float value);
    virtual std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_);
private:
    void* mem_;

    friend struct CPUTensor<DataType::Float>;
    friend struct CPUTensor<DataType::BF16>;
};


} // end of namespace
#endif
