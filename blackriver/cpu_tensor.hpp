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
        } else if ( _DTYPE_ == DataType::Int ) {
            mem_ = MemoryContext::alloc(shape.numel() * sizeof(int));
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
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;
    ComputingReturn io_mpi_bcast(tensor_t self, int root) override;
    ComputingReturn io_mpi_recv(tensor_t self, int source) override;

    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_copy(tensor_t self, tensor_t dst) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) override;
private:
    void* mem_;

    friend struct CPUTensor<DataType::Float>;
    friend struct CPUTensor<DataType::BF16>;
    friend struct CPUTensor<DataType::Int>;
};


} // end of namespace
#endif
