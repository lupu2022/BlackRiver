#include <iostream>
#include <chrono>
#include <unistd.h>

#include "common.hpp"
#include "context.hpp"
#include "dag.hpp"
#include "nn.hpp"
#include "tensortype.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

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

void create_dynamic(br::Enviroment* env, size_t batch, size_t tokens) {
    env->change(0);

    size_t hidden_size = env->hash().find_number("HIDDEN_SIZE");
    size_t vocab_size = env->hash().find_number("VOCAB_SIZE");
    size_t heads = env->hash().find_number("HEADS_NUM");
    size_t hhidden = env->hash().find_number("HEAD_HIDDEN");

    size_t pos = 0;
    // xinput in GPU, xinput_ in CPU
    {
        std::stringstream ss;
        ss << "'_var_' @ " << pos << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xinput' !";
        ss << std::endl;

        ss << "'_xinput_' @  0 [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xinput_' !";
        ss << std::endl;

        env->execute( ss.str() );

        pos = pos + batch * tokens * hidden_size;
    }

    // xmask in GPU, xmask_ in CPU
    {
        std::stringstream ss;
        ss << "'_var_' @ " << pos << " [" << batch << " " << tokens << " " << tokens << "] op.view ";
        ss << " 'xmask' !";
        ss << std::endl;

        ss << "'_xmask_' @  0  [" << batch << " " << tokens << " " << tokens << "] op.view ";
        ss << " 'xmask_' !";
        ss << std::endl;

        env->execute( ss.str() );

        pos = pos + batch * tokens * tokens;
    }

    // alibi in GPU
    {
        std::stringstream ss;
        ss << "'_var_' @ " << pos << " [" << batch << " " << heads << "  1 " << tokens << "] op.view ";
        ss << " dup 'alibi' !  op.build_alibi ";
        ss << std::endl;

        env->execute( ss.str() );

        pos = pos + batch * heads * tokens;
    }

    // xa, xb, xc, xd, xe
    // ya, yb, yc, yd, ye
    // za, zb, zc, zd, ze
    {
        size_t pos2 = pos;

        std::stringstream ss;
        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xa' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << heads << " " << hhidden  << "] op.view ";
        ss << " 'ya' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << hhidden  << "] op.view ";
        ss << " 'za' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xb' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << heads << " " << hhidden  << "] op.view ";
        ss << " 'yb' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << hhidden  << "] op.view ";
        ss << " 'zb' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xc' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << heads << " " << hhidden  << "] op.view ";
        ss << " 'yc' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << hhidden  << "] op.view ";
        ss << " 'zc' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xd' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << heads << " " << hhidden  << "] op.view ";
        ss << " 'yd' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << hhidden  << "] op.view ";
        ss << " 'zd' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size << "] op.view ";
        ss << " 'xe' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << heads << " " << hhidden  << "] op.view ";
        ss << " 'ye' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << hhidden  << "] op.view ";
        ss << " 'ze' ! ";
        ss << std::endl;

        env->execute( ss.str() );
    }

    // x3a x3b
    {
        int pos2 = pos;
        std::stringstream ss;
        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size*3 << "] op.view ";
        ss << " 'x3a' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size * 3;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size*3 << "] op.view ";
        ss << " 'x3b' ! ";
        ss << std::endl;

        env->execute( ss.str() );
    }

    // x4a x4b
    {
        int pos2 = pos;
        std::stringstream ss;
        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size*4 << "] op.view ";
        ss << " 'x4a' ! ";
        ss << std::endl;

        pos2 = pos2 + batch * tokens * hidden_size * 4;

        ss << "'_var_' @ " << pos2 << " [" << batch << " " << tokens << " " << hidden_size*4 << "] op.view ";
        ss << " 'x4b' ! ";
        ss << std::endl;

        env->execute( ss.str() );
    }

    // large mmeory
    {
        size_t pos2 = pos + batch * tokens * hidden_size * 5;

        std::stringstream ss;
        ss << "'_var_' @ " << pos2 << " [" << batch << " " << heads << " " << tokens << " " << tokens << " ] op.view ";
        ss << " 'xll' ! ";
        ss << std::endl;

        ss << "'_var_' @ " << pos2 << " [" << vocab_size << " " << hidden_size << " ] op.view ";
        ss << " 'lm_head.weight' ! ";
        ss << std::endl;

        env->execute( ss.str() );
    }
}

br::Enviroment* create_env(const std::vector<std::string>& layers, bool withOutput) {
    br::Enviroment* env = new br::Enviroment(layers.size() + 1);
    br::load_nn_words(*env);

    std::string init_code = fileToString("model/init.words");
    br::DaG* init_wd = env->build(init_code);

    // 0th hash  is used in GPU
    env->change(0);
    env->hash().set("$DEVICE", "cuda");
    env->run(init_wd);
    env->execute("create_weight");
    env->execute("create_grad");

    size_t total_var = 1024l * 1024 * 1024 * 2 + 1024l*1024*256;
    size_t max_input = 16l * 2048 * 4096;
    size_t max_masks = 16l * 2048 * 2048;
    std::stringstream ss;
    ss << total_var << " " <<  max_input << " " << max_masks << " create_var";
    env->execute( ss.str() );

    // others is used in CPU
    for (size_t i = 1; i < env->hashes_num(); i++) {
        env->change(i);
        env->hash().set("$DEVICE", "cpu");
        env->run( init_wd );
        env->execute("create_weight");
        env->execute("create_grad");

        std::stringstream ss;
        ss << "'" + layers[i-1] + "' load_weight";
        env->execute( ss.str() );
        env->execute( "zero_grad" );

        if ( withOutput && (i == layers.size()) ) {
            env->execute("create_output");
            env->execute("load_output");
        }


    }

    delete init_wd;
    return env;
}

const size_t MEM_CTX_SIZE = 24l * 1024 * 1024 * 1024;

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        std::vector<float> xinput;
        br::read_data("model/xinput.bin", xinput);

        for (int i = 0; i < 2; i++) {
            MPI_Send(xinput.data(), xinput.size(), MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        }

    } else if ( br::CollectiveContext::mpi_rank == 1) {
        br::MemoryContext::boot( MEM_CTX_SIZE + 7l*1024*1024*1024 );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        std::vector<std::string> layers{"h0", "h2", "h4", "h6", "h8", "h10", "h12", "h14", "h16", "h18", "h20", "h22", "h24", "h26", "h28"};
        br::Enviroment* env = create_env(layers, true);

        create_dynamic(env, 40, 512);

        std::string train_code = fileToString("model/train.words");
        env->execute(train_code);

        sleep(15);
        auto start = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < 2; i++) {
            env->execute("train_0");
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time: " << duration.count() << std::endl;

        sleep(5);
        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        std::vector<std::string> layers{"h1", "h3", "h5", "h7", "h9", "h11", "h13", "h15", "h17", "h19", "h21", "h23", "h25", "h27", "h29"};
        br::Enviroment* env = create_env(layers, false);

        std::string train_code = fileToString("model/train.words");
        env->execute(train_code);

        create_dynamic(env, 40, 512);

        sleep(15);
        for ( int i = 0; i < 2; i++) {
            env->execute("train_1");
        }

        sleep(5);
        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 3) {



    } else {
        br_panic("Can't be here!");
    }

    br::CollectiveContext::shutdown();
}

