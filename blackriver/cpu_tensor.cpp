#include <chrono>
#include <msgpack.hpp>

#include "common.hpp"
#include "context.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

namespace br {

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

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_zero(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        memset( mem_, 0, sizeof(float) * self->items() );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_copy(tensor_t self, tensor_t src) {
    if ( _DTYPE_ == DataType::Float ) {
        if ( src->is_cpu() ) {
            memcpy(data(), src->cpu_float()->data(), self->items() * sizeof(float) );
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_float()->data(), self->items() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_fill(tensor_t self, float value) {
    if ( _DTYPE_ == DataType::Float ) {
        float* dst = (float *)data();
        for (size_t i = 0; i < self->items(); i++) {
            dst[i] = value;
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> CPUTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new CPUTensor<DataType::Float>(newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_dump(tensor_t self) {
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

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    // TODO
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
    if ( _DTYPE_ == DataType::Float ) {
        std::vector<float> src;
        load_data(fileName, src);

        br_assert(src.size() == self->items() , "loaded data must has same size");
        memcpy(data(), src.data(), self->items() * sizeof(float));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


tensor_t create_cpu_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CPUTensor<DataType::Float>* tensor = new CPUTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
