#include <cmath>
#include <algorithm>
#include <openblas/cblas.h>

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

        size_t batch = self->shape()[0];
        size_t tokens = self->shape()[1];
        size_t inSize = self->shape()[2];
        size_t outSize = w_->shape()[0];

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
                ComputingContext::workspace_size);

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
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->cuda_float();
        auto v = var->cuda_float();
        auto s = scale->cuda_float();
        auto b = bias->cuda_float();
        auto out = y->cuda_float();

        auto stream = ComputingContext::cuda_stream;
        kernels::launch_layer_norm_float((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, eps, stream);

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_transpos_0213(tensor_t self, tensor_t y) {
    if ( DT == DataType::Float ) {
        auto x = this;

        int sz0 = self->shape()[0];
        int sz1 = self->shape()[1];
        int sz2 = self->shape()[2];
        int sz3 = self->shape()[3];

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
                    ComputingContext::workspace_size);
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
                ComputingContext::workspace_size);
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
        int hhidden = value_->shape()[3];

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
                    ComputingContext::workspace_size);
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
        int batch = self->shape()[0];
        int tokens = self->shape()[1];
        int hidden_size = self->shape()[2];

        int vocab_size = lm_head->shape()[0];

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
                    ComputingContext::workspace_size);
            }
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

/*
// A working reduce API caller
//
// select max value of softmax
cudnnReduceTensorDescriptor_t reduceDesc;

CUDNN_CHECK( cudnnCreateReduceTensorDescriptor(&reduceDesc) );
CUDNN_CHECK( cudnnSetReduceTensorDescriptor(reduceDesc,
                                            CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES) );

int split = n + 1024;
split = split - split % 1024;
float* c = (float *) br::ComputingContext::cuda_workspace;
int* ind = (int *)(c + split);
float* w = c + split * 2;
size_t isize = split * sizeof(int);
size_t wsize = br::ComputingContext::workspace_size - split * 2 * sizeof(float);

auto  cdesc = create_cudnn_td_with({(size_t)n, 1, 1, 1});
CUDNN_CHECK( cudnnReduceTensor(ComputingContext::cudnn_handle,
                                reduceDesc,
                                ind, isize , w, wsize,
                                &alpha, xdesc, dst, &beta, cdesc, c) );

CUDNN_CHECK( cudnnDestroyReduceTensorDescriptor(reduceDesc) );

std::vector<int> ind_;
ind_.resize(split, 0);
CUDA_CHECK(cudaMemcpyAsync(ind_.data(), ind, ind_.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
CUDA_CHECK(cudaStreamSynchronize(stream));
*/

template<DataType DT>
std::variant<ComputingReturn, float> CUDATensor<DT>::op_loss_backward(tensor_t self, tensor_t ids_, tensor_t mask_, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) {
    struct _ {
        static void vocab_embedding(int vsize, int gsize, int hsize, float* lm_head, float* x, float* dst) {
            // computing vocab embedding
            int m = vsize;
            int n = gsize;
            int k = hsize;

            float alpha = 1.0;
            float beta = 0.0;

            float* A = lm_head;
            float* B = x;
            float* C = dst;

            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            m, n, k,
                            &alpha, A, k,
                            B, k, &beta,
                            C, m,
                            ComputingContext::cuda_workspace,
                            ComputingContext::workspace_size);

        }

        static void vocab_grad_x(int vsize, int gsize, int hsize, float* lm_head, float* dx, float* dst) {
            // computing vocab embedding
            int m = hsize;
            int n = gsize;
            int k = vsize;

            float alpha = 1.0;
            float beta = 0;

            float* A = lm_head;
            float* B = dst;
            float* C = dx;

            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k,
                            &alpha, A, m,
                            B, k, &beta,
                            C, m,
                            ComputingContext::cuda_workspace,
                            ComputingContext::workspace_size);

        }

        static void vocab_grad_w(int vsize, int gsize, int hsize, float* lm_head_w, float* x, float* dout) {
            // GEMM in CPU
            int m = hsize;
            int n = vsize;
            int k = gsize;

            float alpha = 1.0;
            float beta = 1.0;

            float* A = x;
            float* B = dout;
            float* C = lm_head_w;

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        m, n, k,
                        alpha, A, m,
                        B, n, beta,
                        C, m);
        }

        static float logits_loss(int vsize, int gsize, float loss_scale,  const int* id, float* logsoftmax, float* dout) {
            int split = gsize + 1024;
            split = split - split % 1024;
            br_assert( split * 2 * sizeof(float) < br::ComputingContext::workspace_size, " workspace size is too small!" );

            auto stream = ComputingContext::cuda_stream;
            int* id_ = (int *) br::ComputingContext::cuda_workspace;
            float* out_ = (float *)id_ + split;

            CUDA_CHECK(cudaMemcpyAsync(id_, id, gsize*sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK( cudaMemset(dout, 0, gsize * vsize * sizeof(float)) );
            kernels::nllloss_forward(id_, logsoftmax, out_, dout, gsize, vsize, loss_scale, stream);

            float sum_loss;
            CUDA_CHECK(cudaMemcpyAsync(&sum_loss, out_, sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            return sum_loss;
        }

        static int count_loss_tokens(int batch, int tokens, const int* mask) {
            int loss_items = 0;
            for (int b = 0;  b < batch; b++) {
                const int* m = &mask[b * tokens];

                for(int t = 0; t < tokens - 1; t++) {
                    if ( m[t] != 0 && m[t+1] != 0) {
                        loss_items ++;
                    }
                }
            }

            return loss_items;
        }
    };

    if ( DT == DataType::Float ) {

        const int batch = self->shape()[0];
        const int tokens = self->shape()[1];
        const int hidden_size = self->shape()[2];

        const int vocab_size = lm_head->shape()[0];

        const int group_max_size = br::ComputingContext::workspace_size / ( vocab_size * sizeof(float) );
        br_assert(  (int)all_logits->items() / vocab_size > 2 * group_max_size, " logits workspace is not enough !");

        int* mask = (int *)mask_->cpu_int()->data();
        int* ids = (int *)ids_->cpu_int()->data();

        double total_loss = 0.0;
        int total_items = _::count_loss_tokens(batch, tokens, mask);
        float loss_scale = 1.0 / (float)total_items;

        float* local_x = new float[self->items()];
        CUDA_CHECK(cudaMemcpyAsync(local_x, data(), self->items() * sizeof(float), cudaMemcpyDeviceToHost, br::ComputingContext::cuda_stream));

        x_g->op_zero(x_g);
        for (int b = 0;  b < batch; b++) {
            const int* m = &mask[b * tokens];
            const int* id = &ids[b * tokens];

            std::vector<int> id_group;
            for(int t = 0; t < tokens - 1; t++) {
                id_group.push_back(t);

                if ( t == (tokens - 2) || (int)id_group.size() == group_max_size ) {
                    // droped last continued masked tokens
                    while( id_group.size() > 0 ) {
                        if ( m[ id_group.back() ] == 0 ) {
                            id_group.pop_back();
                        } else {
                            break;
                        }
                    }
                    if ( id_group.size() == 0 ) {
                        continue;
                    }

                    int begin_t = id_group[0];
                    int loss_items = 0;
                    for(size_t i = 0; i < id_group.size(); i++) {
                        int tt = id_group[i];

                        if ( m[tt] == 0 || m[tt+1] == 0) {
                            id_group[i] = -100;
                        } else {
                            id_group[i] = id[tt+1];
                            loss_items ++;
                        }
                    }

                    if ( loss_items >  0) {
                        float* x = (float *)data() + b * tokens * hidden_size + begin_t * hidden_size;
                        float* dx = (float *)x_g->cuda_float()->data() + b * tokens * hidden_size + begin_t * hidden_size;
                        float* lm = (float *)lm_head->cuda_float()->data();

                        float* x_ = local_x + b * tokens * hidden_size + begin_t * hidden_size;
                        float* local_dst = (float *)br::ComputingContext::local_workspace;
                        float* dst_a = (float *)all_logits->cuda_float()->data();
                        float* dst_b = dst_a + id_group.size() * vocab_size;

                        // do vocab embedding
                        _::vocab_embedding(vocab_size, id_group.size(), hidden_size,
                                           lm, x, dst_a);

                        // do log softmax
                        {
                            float alpha = 1.0;
                            float beta = 0.0;
                            auto xdesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto ydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                            CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                                            &alpha, xdesc, dst_a, &beta, ydesc, dst_a) );
                        }

                        // computing final loss
                        float loss = _::logits_loss(vocab_size, id_group.size(), loss_scale, id_group.data(), dst_a, dst_b);
                        total_loss += loss;


                        // log softmax backward
                        {
                            float alpha = 1.0;
                            float beta = 0.0;
                            auto ydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto dxdesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto dydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});

                            CUDNN_CHECK( cudnnSoftmaxBackward( ComputingContext::cudnn_handle,
                                            CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                                            &alpha, ydesc, dst_a, dydesc, dst_b, &beta, dxdesc, dst_b) );


                            CUDA_CHECK(cudaMemcpyAsync(local_dst, dst_b, id_group.size() * vocab_size * sizeof(float), cudaMemcpyDeviceToHost, br::ComputingContext::cuda_stream));
                        }

                        // computing x_grad
                        _::vocab_grad_x(vocab_size, id_group.size(), hidden_size,
                                        lm, dx, dst_b);


                        // at local (CPU), computing grad for lm_head's weight
                        CUDA_CHECK(cudaStreamSynchronize( br::ComputingContext::cuda_stream ) );
                        _::vocab_grad_w(vocab_size, id_group.size(), hidden_size,
                                        (float *)lm_head_g->cpu_float()->data(), x_, local_dst);

                    }
                    id_group.clear();
                }
            }
        }

        delete local_x;

        float ret = total_loss / total_items;
        std::cout << "#################  " << total_items << " "  <<  ret << std::endl;

        return ret;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_layernorm_backward(tensor_t self, tensor_t scale_, tensor_t bias_, tensor_t var_, tensor_t y_, tensor_t dscale_, tensor_t dbias_, tensor_t din_, float eps) {
    if ( DT == DataType::Float ) {
        cudaStream_t streams[] = {br::ComputingContext::cuda_stream, br::ComputingContext::cuda_stream};

        float* dout = (float *)self->cuda_float()->data();
        float* scale = (float *)scale_->cuda_float()->data();
        float* bias = (float *)bias_->cuda_float()->data();
        float* var = (float *)var_->cuda_float()->data();
        float* y = (float *)y_->cuda_float()->data();
        float* dscale = (float *)dscale_->cuda_float()->data();
        float* dbias = (float *)dbias_->cuda_float()->data();
        float* din = (float *)din_->cuda_float()->data();

        size_t batch = self->shape()[0];
        size_t tokens = self->shape()[1];
        int hidden = self->shape()[2];
        int num = batch * tokens;

        kernels::launch_ln_bw_float(dscale, dbias, din, dout,
                                    nullptr, y, scale, bias,
                                    var, nullptr,
                                    num, hidden, eps, streams);


        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) {
    if ( DT == DataType::Float ) {
        size_t batch = self->shape()[0];
        size_t tokens = self->shape()[1];
        size_t outSize = self->shape()[2];
        size_t inSize = x->shape()[2];

        // do reduce follow batch
        {
            cudnnReduceTensorDescriptor_t reduceDesc;

            CUDNN_CHECK( cudnnCreateReduceTensorDescriptor(&reduceDesc) );
            CUDNN_CHECK( cudnnSetReduceTensorDescriptor(reduceDesc,
                                                        CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                                        CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES) );

            float alpha = 1.0;
            float beta = 0.0;
            auto  adesc = create_cudnn_td_with({batch * tokens, outSize, 1, 1});
            auto  cdesc = create_cudnn_td_with({1,              outSize, 1, 1});
            void* a = data();
            void* c = bias_g->cuda_float()->data();

            CUDNN_CHECK( cudnnReduceTensor(ComputingContext::cudnn_handle,
                                reduceDesc,
                                nullptr, 0,
                                br::ComputingContext::cuda_workspace, br::ComputingContext::workspace_size,
                                &alpha, adesc, a, &beta, cdesc, c) );

            CUDNN_CHECK( cudnnDestroyReduceTensorDescriptor(reduceDesc) );
        }

        // do twice gemm
        {
            int m = inSize;
            int n = outSize;
            int k = batch * tokens;

            float* A = (float *)x->cuda_float()->data();
            float* B = (float *)data();
            float* C = (float *)weight_g->cuda_float()->data();

            float alpha = 1.0;
            float beta = 0.0;

            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha, A, m,
                    B, n, &beta,
                    C, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }

        {
            int m = inSize;
            int n = batch * tokens;
            int k = outSize;

            float* A = (float *)weight->cuda_float()->data();
            float* B = (float *)data();
            float* C = (float *)x_g->cuda_float()->data();

            float alpha = 1.0;
            float beta = 0.0;

            kernels::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, m,
                    B, k, &beta,
                    C, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }


        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_gelu_backward(tensor_t self, tensor_t x_, tensor_t x_g_) {
    if ( DT == DataType::Float ) {
        auto stream = ComputingContext::cuda_stream;
        float* out_g = (float *)data();
        float* x = (float *)x_->cuda_float()->data();
        float* x_g = (float *)x_g_->cuda_float()->data();

        kernels::gelu_backward(out_g, x, x_g, self->items(), stream);


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
