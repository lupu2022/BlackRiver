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

const size_t MEM_CTX_SIZE = 25.5 * 1024 * 1024 * 1024l;
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
        const size_t batch = 2;
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
        env_->run(init_wd);
        if ( br::CollectiveContext::mpi_rank == 1 ) {
            env_->execute("create_main_weight");
            env_->execute("create_main_grad");
            env_->execute("load_main_weight");
            env_->execute("zero_main_grad");

        } else if ( br::CollectiveContext::mpi_rank == 2) {
            env_->execute(" 'cuda' $DEVICE ! create_layer_weight create_layer_grad ");
        }

        // others is used in CPU
        for (size_t i = 1; i < env_->hashes_num(); i++) {
            env_->change(i);
            env_->hash().set("$DEVICE", "cpu");
            env_->run( init_wd );
            env_->execute("create_layer_input");
            env_->execute("create_layer_weight");
            env_->execute("create_layer_grad");
            std::stringstream ss;
            ss << "'" << layers_[i-1] << "' load_layer_weight";
            env_->execute( ss.str() );
            env_->execute("zero_layer_grad");
        }
        delete init_wd;

        std::string train_code = fileToString("model/train.words");
        env_->change(0);
        env_->execute(train_code);

        size_t total_var = 1024l * 1024 * 1024 * 2 + 1024l*1024*256;
        std::stringstream ss;
        ss << total_var << " create_var";
        env_->execute( ss.str() );
    }

    ~BloomAttentions() {
        delete env_;
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

        std::stringstream ss;
        ss << batch << " " << tokens << " create_dynamic";
        env_->execute( ss.str() );

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
        br::MemoryContext::boot( MEM_CTX_SIZE + 7l*1024*1024*1024);
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

    br::CollectiveContext::shutdown();
}

