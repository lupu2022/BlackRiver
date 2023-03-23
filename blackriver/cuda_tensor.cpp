#include <cmath>
#include <algorithm>

#include "context.hpp"
#include "cuda_tensor.hpp"
#include "cpu_tensor.hpp"
#include "kernels/kernels.h"

namespace br {

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);

    if ( DT == DataType::Float ) {
        auto stream = ComputingContext::cuda_stream;
        std::vector<float> local_first;
        std::vector<float> local_last;

        local_first.resize(first8, 0);
        local_last.resize(first8, 0);

        auto x = self->cuda_float();
        CUDA_CHECK(cudaMemcpyAsync(local_first.data(), x->data(), local_first.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        std::vector<size_t> pos = self->shape().vec();
        auto shape_ = self->shape().vec();
        for(int i = 0; i < (int)pos.size() - 1; i++) {
            pos[i] = shape_[i] - 1;
        }
        pos.back() = shape_.back() - first8;
        void* src = (float *)x->data() + self->items() - first8;
        CUDA_CHECK(cudaMemcpyAsync(local_last.data(), src, local_last.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "--------------------------" << std::endl;
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_first[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_last[i] << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_zero(tensor_t self) {
    if ( DT == DataType::Float ) {
        void *dst = data();
        int n = self->items();
        CUDA_CHECK( cudaMemset(dst, 0, n * sizeof(float)) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_fill(tensor_t self, float value) {
    if ( DT == DataType::Float ) {
        float* dst = (float *)data();
        auto desc = create_cudnn_td_with( {self->items()} );
        CUDNN_CHECK( cudnnSetTensor( ComputingContext::cudnn_handle,
                                     desc,
                                     dst,
                                     &value) );

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_copy(tensor_t self, tensor_t src) {
    if ( DT == DataType::Float ) {
        if ( src->is_cpu() ) {
            void* x = src->cpu_float()->data();
            void* y = data();

            CUBLAS_CHECK( cublasSetVector(self->items(), sizeof(float), x, 1, y, 1) );
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_float()->data(), self->items() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear(tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    if ( DT == DataType::Float ) {
        auto x = this;
        auto w = w_->cuda_float();
        auto b = b_->cuda_float();
        auto y = y_->cuda_float();

        size_t batch = self->shape().vec()[0];
        size_t tokens = self->shape().vec()[1];
        size_t inSize = self->shape().vec()[2];
        size_t outSize = w_->shape().vec()[0];

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        float* A = (float *)w->data();
        float* B = (float *)x->data();
        float* C = (float *)y->data();
        void* bias = b->data();

        float alpha = 1.0;
        float beta = 0.0;

        kernels::LtSgemm(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, k,
                B, k, &beta,
                C, m,
                ComputingContext::cuda_workspace,
                ComputingContext::cuda_workspace_size);

        {
            auto ydesc = y->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b->create_cudnn_td_with({1, 1, 1, outSize});

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t> CUDATensor<DT>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( DT == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCudaTensor = new CUDATensor<DataType::Float>(newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_build_alibi(tensor_t self) {
    if ( DT == DataType::Float ) {
        size_t batch = self->shape().vec()[0];
        size_t heads = self->shape().vec()[1];
        size_t tokens = self->shape().vec()[2];

        if ( heads & ( heads - 1) ) {
            br_panic("Only support heads number is power of 2");
        }

        double base = 3 - log2(heads*1.0);
        base = -1 * pow(2.0, base);
        base = pow(2.0, base);

        std::vector<float> ldata;
        for (int i = 0; i < (int)batch; i++) {
            for (int j = 0; j < (int)heads; j++) {
                double slope = pow(base, (j + 1) * 1.0);
                for (int k = 0; k < (int)tokens; k++) {
                    ldata.push_back( k * 1.0 * slope );
                }
            }
        }

        CUBLAS_CHECK( cublasSetVector(self->items(), sizeof(float), ldata.data(), 1, data() , 1) );

        return OP_OK;
    }
    return OP_TODO_ERROR;

}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    if ( DT == DataType::Float ) {
        auto adesc = create_cudnn_td_with( self->shape().vec() );
        auto bdesc = b->cuda_float()->create_cudnn_td_with( b->shape().vec() );
        auto cdesc = c->cuda_float()->create_cudnn_td_with( c->shape().vec() );

        float alpha = 1.0;
        float beta = 0.0;

        cudnnOpTensorDescriptor_t opTensorDesc;
        CUDNN_CHECK( cudnnCreateOpTensorDescriptor(&opTensorDesc) );
        CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );

        CUDNN_CHECK( cudnnOpTensor(ComputingContext::cudnn_handle,
                                    opTensorDesc,
                                    &alpha, adesc, data(),
                                    &alpha, bdesc, b->cuda_float()->data(),
                                    &beta,  cdesc, c->cuda_float()->data()) );

        CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );

        return OP_OK;
    }

    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn CUDATensor<DT>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y) {
    if ( DT == DataType::Float ) {
        auto x = this;
        size_t batch = self->shape().vec()[0] * self->shape().vec()[1];
        size_t hidden = self->shape().vec()[2];

        auto m = mean->cuda_float();
        auto v = var->cuda_float();
        auto s = scale->cuda_float();
        auto b = bias->cuda_float();
        auto out = y->cuda_float();

        auto stream = ComputingContext::cuda_stream;
        kernels::launch_layer_norm_float((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, stream);

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_transpos_0213(tensor_t self, tensor_t y) {
    if ( DT == DataType::Float ) {
        auto x = this;

        int sz0 = self->shape().vec()[0];
        int sz1 = self->shape().vec()[1];
        int sz2 = self->shape().vec()[2];
        int sz3 = self->shape().vec()[3];

        auto out = y->cuda_float();

        auto stream = ComputingContext::cuda_stream;
        kernels::launch_transform_0213<float>((float *)x->data(), (float *)out->data(), sz0, sz1, sz2, sz3, stream);

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_qk(tensor_t self, tensor_t k_, tensor_t qk_) {
    if ( DT == DataType::Float ) {

        auto shape_ = self->shape().vec();

        int batch = shape_[0];
        int heads = shape_[1];
        int tokens = shape_[2];
        int hhidden = shape_[3];

        int m = tokens;
        int n = tokens;
        int k = hhidden;

        float alpha = 1.0 / sqrt(hhidden);
        float beta = 0.0;

#if 1
        int HT = hhidden * tokens ;
        int TT = tokens * tokens;

        for (int i = 0; i < batch * heads; i++) {
            float* B = (float *)data() + i * HT;
            float* A = (float *)(k_->cuda_float()->data()) + i * HT;
            float* C = (float *)(qk_->cuda_float()->data()) + i * TT;
            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, k,
                    B, k, &beta,
                    C, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::cuda_workspace_size);
        }
#else
        float* B = (float *)data();
        float* A = (float *)(k_->cuda_float()->data());
        float* C = (float *)(qk_->cuda_float()->data());
        kernels::LtSgemmBatched(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, k,
                B, k, &beta,
                C, m,
                batch * heads,
                ComputingContext::cuda_workspace,
                ComputingContext::cuda_workspace_size);
#endif

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_softmax(tensor_t self, tensor_t y) {
    if ( DT == DataType::Float ) {
        float alpha = 1.0;
        float beta = 0.0;

        auto shape_ = self->shape().vec();

        size_t batch = shape_[0];
        size_t heads = shape_[1];
        size_t tokens = shape_[2];

        void* xdata = data();
        void* ydata = y->cuda_float()->data();

        auto xdesc = create_cudnn_td_with({ batch * heads * tokens, tokens, 1, 1});
        auto ydesc = create_cudnn_td_with({ batch * heads * tokens, tokens, 1, 1});
        CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                          CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                          &alpha, xdesc, xdata, &beta, ydesc, ydata) );

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_attn(tensor_t self, tensor_t value_, tensor_t out_) {
    if ( DT == DataType::Float ) {
        float alpha = 1.0;
        float beta = 0.0;

        auto value = value_->cuda_float();
        auto out = out_->cuda_float();

        auto shape_ = self->shape().vec();

        int batch = shape_[0];
        int heads = shape_[1];
        int tokens = shape_[2];
        int hhidden = value_->shape().vec()[3];

        int m = hhidden;
        int n = tokens;
        int k = tokens;

        int HT = hhidden * tokens ;
        int TT = tokens * tokens;
        for (int i = 0; i < batch * heads; i++) {
            float* A = (float *)(value->data()) + i * HT;
            float* B = (float *)data() + i * TT;
            float* C = (float *)(out->data()) + i * HT;

            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, m,
                    B, k, &beta,
                    C, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::cuda_workspace_size);
        }

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_gelu(tensor_t self, tensor_t out) {
    if ( DT == DataType::Float ) {
        auto stream = ComputingContext::cuda_stream;
        float* src = (float *)data();
        float* dst = (float *)out->cuda_float()->data();

        kernels::gelu_forward(src, dst, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

tensor_t create_cuda_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Float>* tensor = new CUDATensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

} // end of namespace br
