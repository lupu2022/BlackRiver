#include "tensortype.hpp"
#include "context.hpp"
#include "nn.hpp"

namespace br {

namespace nn {
    std::vector<size_t> fetch_shape(Stack& stack) {
        auto nums = stack.pop_number_list();
        std::vector<size_t> shape;
        for ( size_t i = 0; i < nums.size(); i++) {
            shape.push_back( nums[i] );
        }
        return shape;
    }
    struct Sync : public NativeWord {
        virtual void run(Stack& stack) {
            auto stream = br::ComputingContext::cuda_stream;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        NWORD_CREATOR_DEFINE_LR(Sync)
    };
    struct Create : public NativeWord {
        virtual void run(Stack& stack) {
            auto device = stack.pop_string();
            auto shape = fetch_shape(stack);
            tensor_t t;
            if ( device == "cuda" ) {
                t = br::create_cuda_float(shape);
            } else if ( device == "cpu" ) {
                t = br::create_cpu_float(shape);
            } else {
                br_panic("Can' be here!");
            }
            stack.push_tensor(t);
        }
        NWORD_CREATOR_DEFINE_LR(Create)
    };

    struct Zero : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t t = stack.pop_tensor();
            t->op_zero(t);
        }
        NWORD_CREATOR_DEFINE_LR(Zero)
    };

    struct Fill : public NativeWord {
        virtual void run(Stack& stack) {
            double value = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_fill(t, value);
        }
        NWORD_CREATOR_DEFINE_LR(Fill)
    };

    struct View : public NativeWord {
        virtual void run(Stack& stack) {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view(t, offset, shape);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(View)
    };

    struct Copy : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_copy(dst, src);
        }
        NWORD_CREATOR_DEFINE_LR(Copy)
    };

    struct Linear : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t y = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t w = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_linear(x, w, b, y);
        }
        NWORD_CREATOR_DEFINE_LR(Linear)
    };

    struct Layernorm : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t y = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t var = stack.pop_tensor();
            tensor_t mean = stack.pop_tensor();

            tensor_t x = stack.pop_tensor();

            x->op_layernorm(x, mean, var, scale, bias, y);
        }
        NWORD_CREATOR_DEFINE_LR(Layernorm)
    };

    struct Transpos0213 : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t y = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpos_0213(x,y);
        }
        NWORD_CREATOR_DEFINE_LR(Transpos0213)
    };

    struct QueryKey : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t qk = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();

            q->op_qk(q, k, qk);
        }
        NWORD_CREATOR_DEFINE_LR(QueryKey)
    };

    struct BuildAlibi : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t alibi = stack.pop_tensor();
            alibi->op_build_alibi(alibi);
        }
        NWORD_CREATOR_DEFINE_LR(BuildAlibi)
    };

    struct Add : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_add(x, b, c);
        }
        NWORD_CREATOR_DEFINE_LR(Add)
    };

    struct Softmax : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_softmax(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Softmax)
    };

    struct Attn : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t out = stack.pop_tensor();
            tensor_t value = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_attn(x, value, out);
        }
        NWORD_CREATOR_DEFINE_LR(Attn)
    };

    struct Gelu : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_gelu(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Gelu)
    };

}

namespace io {
    struct Dump : public NativeWord {
        virtual void run(Stack& stack) {
            tensor_t t = stack.pop_tensor();
            t->io_dump(t);
        }
        NWORD_CREATOR_DEFINE_LR(Dump)
    };

    struct Load : public NativeWord {
        virtual void run(Stack& stack) {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_load(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Load)
    };

    struct Save : public NativeWord {
        virtual void run(Stack& stack) {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_save(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Save)
    };

    struct MPIRecv : public NativeWord {
        virtual void run(Stack& stack) {
            int source = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_recv(x, source);
        }
        NWORD_CREATOR_DEFINE_LR(MPIRecv)
    };

}

void load_nn_words(Enviroment& env) {
    env.insert_native_word("io.dump", io::Dump::creator );
    env.insert_native_word("io.load", io::Load::creator );
    env.insert_native_word("io.save", io::Save::creator );
    env.insert_native_word("io.mpi.recv", io::MPIRecv::creator );

    env.insert_native_word("op.sync", nn::Sync::creator );
    env.insert_native_word("op.create", nn::Create::creator );
    env.insert_native_word("op.zero", nn::Zero::creator );
    env.insert_native_word("op.fill", nn::Fill::creator );
    env.insert_native_word("op.view", nn::View::creator );
    env.insert_native_word("op.copy", nn::Copy::creator );
    env.insert_native_word("op.linear", nn::Linear::creator );
    env.insert_native_word("op.layernorm", nn::Layernorm::creator );
    env.insert_native_word("op.transpos_0213", nn::Transpos0213::creator );
    env.insert_native_word("op.build_alibi", nn::BuildAlibi::creator);
    env.insert_native_word("op.add", nn::Add::creator);
    env.insert_native_word("op.querykey", nn::QueryKey::creator);
    env.insert_native_word("op.softmax", nn::Softmax::creator);
    env.insert_native_word("op.attn", nn::Attn::creator);
    env.insert_native_word("op.gelu", nn::Gelu::creator);
}

}// end of namespace br
