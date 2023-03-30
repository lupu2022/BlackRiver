#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include "common.hpp"
#include "context.hpp"
#include "dag.hpp"
#include "nn.hpp"
#include "tensortype.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

const size_t MEM_CTX_SIZE = 24l * 1024 * 1024 * 1024;
const int VOCAB_SIZE = 250880;
const int HIDDEN_SIZE = 4096;
const int HEADS_NUM = 32;
const int HEAD_HIDDEN = 128;

inline std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

struct BloomInput {
    BloomInput() {
        br::read_data("model/weights/word_embeddings.weight.bin", vocab);
    }
    ~BloomInput() {
    }

    std::tuple<int, int> fetch_data(std::vector<int>& ids, std::vector<int>& mask, std::vector<float>& xinput, std::vector<float>& alibi, std::vector<float>& xmask) {
        const size_t batch = 4;
        const size_t tokens = 512;

        // fill ids&masks
        br::read_data("model/xinput.ids.bin", ids);
        br::read_data("model/xinput.mask.bin", mask);

        // load embeddings
        xinput.resize(batch * tokens * HIDDEN_SIZE);
        for (size_t i = 0; i < ids.size(); i++) {
            size_t id = ids[i];
            const float* embedding = &vocab[id * HIDDEN_SIZE];
            float* dst = &xinput[i * HIDDEN_SIZE];
            memcpy(dst, embedding, HIDDEN_SIZE * sizeof(float) );
        }

        // building alibi
        alibi.clear();
        {
            double base = 3 - log2(HEADS_NUM*1.0);
            base = -1 * pow(2.0, base);
            base = pow(2.0, base);

            for (int j = 0; j < (int)HEADS_NUM; j++) {
                double slope = pow(base, (j + 1) * 1.0);
                for (int k = 0; k < (int)tokens; k++) {
                    alibi.push_back( k * 1.0 * slope );
                }
            }
        }

        // building xmask
        xmask.resize(1l * batch * tokens * tokens );
        std::fill(xmask.begin(), xmask.end(), -1.0 * std::numeric_limits<float>::max());

        for (size_t i = 0; i < batch; i++) {
            const int* ms = &mask[i * tokens];
            float* m2d = &xmask[ i * tokens * tokens ];

            for (size_t m = 0; m < tokens; m++) {
                for( size_t n = 0; n <= m; n++) {
                    if ( ms[n] != 0) {
                        m2d[m * tokens + n] = 0.0;
                    } else {
                        break;
                    }
                }
            }
        }

        return {batch, tokens};
    }

    void sub_batch() {
        std::vector<int> ids;
        std::vector<int> mask;
        std::vector<float> xinput;
        std::vector<float> alibi;
        std::vector<float> xmask;

        auto ret = fetch_data(ids, mask, xinput, alibi, xmask);
        int batch = std::get<0>(ret);
        int tokens = std::get<1>(ret);

        // broadcast common data
        MPI_Bcast(&batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(ids.data(), ids.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(mask.data(), mask.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(alibi.data(), alibi.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(xmask.data(), xmask.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        // sending to first layer
        MPI_Send(xinput.data(), xinput.size(), MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    }

public:
    std::vector<float> vocab;
};

struct BloomAttentions {
    BloomAttentions(const std::vector<const char*>& layers) : layers_(layers) {
        env_ = new br::Enviroment(layers_.size() + 1);
        br::load_nn_words(*env_);

        std::string init_code = fileToString("model/init.words");
        br::DaG* init_wd = env_->build(init_code);

        // 0th hash  is used in GPU
        env_->change(0);
        env_->hash().set("$DEVICE", "cuda");
        env_->run(init_wd);
        env_->execute("create_weight");
        env_->execute("create_grad");

        size_t total_var = 1024l * 1024 * 1024 * 2 + 1024l*1024*256;
        size_t max_alibi = HEADS_NUM * 2048;
        size_t max_input  = 16l * 2048 * HIDDEN_SIZE;
        size_t max_xmask  = 16l * 2048 * 2048;

        std::stringstream ss;
        ss << total_var << " " <<  max_input << " " << max_xmask << " " << max_alibi << " create_var";
        env_->execute( ss.str() );

        // others is used in CPU
        for (size_t i = 1; i < env_->hashes_num(); i++) {
            env_->change(i);
            env_->hash().set("$DEVICE", "cpu");
            env_->run( init_wd );
            env_->execute("create_weight");
            env_->execute("create_grad");

            std::stringstream ss;
            ss << "'" << layers_[i-1] << "' load_weight";
            env_->execute( ss.str() );
            env_->execute( "zero_grad" );

            if ( (br::CollectiveContext::nccl_rank == 0)  && (i == layers_.size()) ) {
                env_->execute("create_output");
                env_->execute("load_output");
            }
        }
        delete init_wd;

        std::string train_code = fileToString("model/train.words");
        env_->change(0);
        env_->execute(train_code);
    }

    ~BloomAttentions() {
        delete env_;
    }

    void create_dynamic(size_t batch, size_t tokens) {
        env_->change(0);

        size_t pos = 0;
        // xinput in GPU, xinput_ in CPU
        {
            std::stringstream ss;
            ss << "'_var_' @ " << pos << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xinput' !";
            ss << std::endl;

            ss << "'_xinput_' @  0 [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xinput_' !";
            ss << std::endl;

            env_->execute( ss.str() );

            pos = pos + batch * tokens * HIDDEN_SIZE;
        }

        // alibi in GPU, alibi_ in CPU
        {
            std::stringstream ss;
            ss << "'_var_' @ " << pos << " [1 " << HEADS_NUM << " 1 " << tokens << "] op.view ";
            ss << " 'alibi' !";
            ss << std::endl;

            ss << "'_alibi_' @  0  [1 " << HEADS_NUM << " 1 " << tokens << "] op.view ";
            ss << " 'alibi_' !";
            ss << std::endl;

            env_->execute( ss.str() );

            pos = pos + HEADS_NUM * tokens;
        }

        // xmask in GPU, xmask_ in CPU
        {
            std::stringstream ss;
            ss << "'_var_' @ " << pos << " [" << batch << " 1 " << tokens << " " << tokens << "] op.view ";
            ss << " 'xmask' !";
            ss << std::endl;

            ss << "'_xmask_' @  0  [" << batch << " 1 " << tokens << " " << tokens << "] op.view ";
            ss << " 'xmask_' !";
            ss << std::endl;

            env_->execute( ss.str() );

            pos = pos + batch * tokens * tokens;
        }

        /*
        // alibi in GPU
        {
            std::stringstream ss;
            ss << "'_var_' @ " << pos << " [" << batch << " " << HEADS_NUM << "  1 " << tokens << "] op.view ";
            ss << " dup 'alibi' ! op.build_alibi";
            ss << std::endl;

            env_->execute( ss.str() );

            pos = pos + batch * HEADS_NUM * tokens;
        }
        */

        // xa, xb, xc, xd, xe
        // ya, yb, yc, yd, ye
        // za, zb, zc, zd, ze
        {
            size_t pos2 = pos;

            std::stringstream ss;
            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xa' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HEADS_NUM << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'ya' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'za' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xb' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HEADS_NUM << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'yb' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'zb' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xc' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HEADS_NUM << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'yc' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'zc' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xd' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HEADS_NUM << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'yd' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'zd' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE << "] op.view ";
            ss << " 'xe' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HEADS_NUM << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'ye' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << HEAD_HIDDEN  << "] op.view ";
            ss << " 'ze' ! ";
            ss << std::endl;

            env_->execute( ss.str() );
        }

        // x3a x3b
        {
            int pos2 = pos;
            std::stringstream ss;
            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE*3 << "] op.view ";
            ss << " 'x3a' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE * 3;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE*3 << "] op.view ";
            ss << " 'x3b' ! ";
            ss << std::endl;

            env_->execute( ss.str() );
        }

        // x4a x4b
        {
            int pos2 = pos;
            std::stringstream ss;
            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE*4 << "] op.view ";
            ss << " 'x4a' ! ";
            ss << std::endl;

            pos2 = pos2 + batch * tokens * HIDDEN_SIZE * 4;

            ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << HIDDEN_SIZE*4 << "] op.view ";
            ss << " 'x4b' ! ";
            ss << std::endl;

            env_->execute( ss.str() );
        }

        // large mmeory
        {
            size_t pos2 = pos + batch * tokens * HIDDEN_SIZE * 5;

            std::stringstream ss;
            ss << "'_var_' @ " << pos2 << " [" << batch << " " << HEADS_NUM << " " << tokens << " " << tokens << " ] op.view ";
            ss << " 'xll' ! ";
            ss << std::endl;

            ss << "'_var_' @ " << pos2 << " [" << VOCAB_SIZE << " " << HIDDEN_SIZE << " ] op.view ";
            ss << " 'lm_head.weight' ! ";
            ss << std::endl;

            env_->execute( ss.str() );
        }
    }

    void forward_backward() {
        int batch = -1;
        int tokens = -1;
        MPI_Bcast(&batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> ids;
        std::vector<int> mask;
        ids.resize(batch * tokens);
        mask.resize(batch * tokens);

        MPI_Bcast(ids.data(), ids.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(mask.data(), mask.size(), MPI_INT, 0, MPI_COMM_WORLD);

        create_dynamic(batch, tokens);
        if ( br::CollectiveContext::nccl_rank == 0) {
            env_->execute("train_0");
        } else if ( br::CollectiveContext::nccl_rank == 1) {
            env_->execute("train_1");
        }
    }


public:
    const std::vector<const char*> layers_;
    br::Enviroment* env_;
};

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        BloomInput* in = new BloomInput();

        in->sub_batch();

        delete in;
    } else if ( br::CollectiveContext::mpi_rank == 1) {
        br::MemoryContext::boot( MEM_CTX_SIZE + 7l*1024*1024*1024 );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        std::vector<const char*> layers{"h0", "h2", "h4", "h6", "h8", "h10", "h12", "h14", "h16", "h18", "h20", "h22", "h24", "h26", "h28"};
        BloomAttentions* attn = new BloomAttentions(layers);

        /*
        auto start = std::chrono::high_resolution_clock::now();
        attn->do_train(4, 512);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time: " << duration.count() << std::endl;
        */

        attn->forward_backward();


        sleep(5);
        delete attn;

        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        std::vector<const char*> layers{"h1", "h3", "h5", "h7", "h9", "h11", "h13", "h15", "h17", "h19", "h21", "h23", "h25", "h27", "h29"};
        BloomAttentions* attn = new BloomAttentions(layers);

        attn->forward_backward();


        sleep(5);
        delete attn;

        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else {
        br_panic("Can't be here!");
    }

    std::cout << "Bye " << br::CollectiveContext::mpi_rank << std::endl;

    br::CollectiveContext::shutdown();
}

