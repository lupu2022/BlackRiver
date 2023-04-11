#include <cmath>
#include <algorithm>

#include "common.hpp"
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
ComputingReturn CUDATensor<DT>::io_load(tensor_t self, const char* fileName) {
    if ( DT == DataType::Float ) {
        std::vector<float> src;
        read_data(fileName, src);

        br_assert(src.size() == self->items() , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_nccl_send(tensor_t self, int dst) {
    if ( DT == DataType::Float ) {
        NCCL_CHECK( ncclSend(data(), self->items(), ncclFloat32, dst,
                             CollectiveContext::nccl_comm,
                             ComputingContext::cuda_stream) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_nccl_recv(tensor_t self, int dst) {
    if ( DT == DataType::Float ) {
        NCCL_CHECK( ncclRecv(data(), self->items(), ncclFloat32, dst,
                             CollectiveContext::nccl_comm,
                             ComputingContext::cuda_stream) );
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

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, self->items() * sizeof(float), cudaMemcpyHostToDevice, stream));
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
ComputingReturn CUDATensor<DT>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
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

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_last_logits(tensor_t self, tensor_t mask_,  tensor_t lm_head, tensor_t output) {
    if ( DT == DataType::Float ) {
        int batch = self->shape().vec()[0];
        int tokens = self->shape().vec()[1];
        int hidden_size = self->shape().vec()[2];

        int vocab_size = lm_head->shape().vec()[0];

        int* mask = (int *)mask_->cpu_int()->data();
        for (int b = 0;  b < batch; b++) {
            int* m = &mask[b * tokens];
            int target = 0;
            for ( int i = 0; i < tokens - 1; i++) {
                if ( m[i + 1] == 0 ) {
                    target = i;
                    break;
                }
            }

            float* dst = (float *)output->cuda_float()->data() + b * vocab_size;
            float* x = (float *)data() + b * tokens * hidden_size + target * hidden_size;

            {
                int m = vocab_size;
                int n = 1;
                int k = hidden_size;

                float alpha = 1.0;
                float beta = 0.0;

                float* A = (float *)lm_head->cuda_float()->data();
                float* B = x;
                float* C = dst;

                kernels::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, k,
                    B, k, &beta,
                    C, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::cuda_workspace_size);
            }
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
std::variant<ComputingReturn, float> CUDATensor<DT>::op_loss_backward(tensor_t self, tensor_t ids_, tensor_t mask_, tensor_t lm_head, tensor_t workspace, tensor_t x_g, tensor_t lm_head_g) {
    if ( DT == DataType::Float ) {
        int batch = self->shape().vec()[0];
        int tokens = self->shape().vec()[1];
        int hidden_size = self->shape().vec()[2];

        int vocab_size = lm_head->shape().vec()[0];
        size_t wsize = workspace->items();

        int token_group = wsize / vocab_size;

        int* mask = (int *)mask_->cpu_int()->data();
        int* ids = (int *)ids_->cpu_int()->data();

        double total_loss = 0.0;
        int total_items = 0;

        auto stream = ComputingContext::cuda_stream;
        for (int b = 0;  b < batch; b++) {
            int* m = &mask[b * tokens];
            int* id = &ids[b * tokens];

            std::vector<int> groups;
            for (int t = 0; t < tokens - 1; t++) {
                bool do_logits = false;
                if ( m[t] == 0) {
                    do_logits = true;
                } else {
                    groups.push_back(t);
                    if ( groups.size() == token_group ) {
                        do_logits = true;
                    }
                }
                if ( t == tokens - 2 ) {
                    do_logits = true;
                }
                if ( do_logits && groups.size() > 0 ) {

                    float* dst = (float *)workspace->cuda_float()->data();
                    float* x = (float *)data() + b * tokens * hidden_size + groups[0] * hidden_size;

                    {
                        int m = vocab_size;
                        int n = groups.size();
                        int k = hidden_size;

                        float alpha = 1.0;
                        float beta = 0.0;

                        float* A = (float *)lm_head->cuda_float()->data();
                        float* B = x;
                        float* C = dst;

                        kernels::LtSgemm(ComputingContext::cublasLt_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            m, n, k,
                            &alpha, A, k,
                            B, k, &beta,
                            C, m,
                            ComputingContext::cuda_workspace,
                            ComputingContext::cuda_workspace_size);

                        auto xdesc = create_cudnn_td_with({ (size_t)n, (size_t)vocab_size, 1, 1});
                        auto ydesc = create_cudnn_td_with({ (size_t)n, (size_t)vocab_size, 1, 1});
                        CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                          CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                                          &alpha, xdesc, dst, &beta, ydesc, dst) );

                        /*
                        // select max value of softmax
                        cudnnReduceTensorDescriptor_t reduceDesc;

                        CUDNN_CHECK( cudnnCreateReduceTensorDescriptor(&reduceDesc) );
                        CUDNN_CHECK( cudnnSetReduceTensorDescriptor(reduceDesc,
                                                                    CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT,
                                                                    CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES) );

                        float* c = (float *) br::ComputingContext::cuda_workspace;
                        int* ind = (int *)(c + 2048);
                        float* w = c + 4096;
                        size_t isize = 2048 * sizeof(int);
                        size_t wsize = br::ComputingContext::cuda_workspace_size - 4096 * sizeof(float);

                        auto  cdesc = create_cudnn_td_with({(size_t)n, 1, 1, 1});
                        CUDNN_CHECK( cudnnReduceTensor(ComputingContext::cudnn_handle,
                                                       reduceDesc,
                                                       ind, isize , w, wsize,
                                                       &alpha, xdesc, dst, &beta, cdesc, c) );

                        CUDNN_CHECK( cudnnDestroyReduceTensorDescriptor(reduceDesc) );
                        */
                    }
                    groups.clear();
                }
            }
        }

        return OP_OK;
    }
    return OP_TODO_ERROR;
}


tensor_t create_cuda_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Float>* tensor = new CUDATensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_half(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::BF16>* tensor = new CUDATensor<DataType::BF16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}


} // end of namespace br
