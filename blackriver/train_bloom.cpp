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

void init_env(br::Enviroment* env) {
    std::string init_code = fileToString("model/init.words");
    br::DaG* init_wd = env->build(init_code);

    // 0th hash  is used in GPU
    env->change(0);
    env->hash().set("$DEVICE", "cuda");
    env->run(init_wd);
    env->execute("create_weight");
    env->execute("create_grad");
    env->execute("create_var");

    // others is used in CPU
    for (size_t i = 1; i < env->hashes_num(); i++) {
        env->change(i);
        env->hash().set("$DEVICE", "cpu");
        env->run( init_wd );
        env->execute("create_weight");
        env->execute("create_grad");
    }

    delete init_wd;
}

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        std::vector<float> xinput;
        br::load_data("model/xinput.msg", xinput);

        sleep(30);

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << " ###################### " << std::endl;
        MPI_Send(xinput.data(), xinput.size(), MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if ( br::CollectiveContext::mpi_rank == 1) {
        //br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(15 + 1);
        br::load_nn_words(*env);

        init_env(env);

        std::string train_code = fileToString("model/train.words");
        env->execute(train_code);
        env->execute("train_0");

        sleep(5);

        delete env;

        br::ComputingContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        //br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(15 + 1);
        br::load_nn_words(*env);

        init_env(env);

        std::string train_code = fileToString("model/train.words");
        env->execute(train_code);
        env->execute("train_1");

        sleep(5);

        delete env;

        br::ComputingContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 3) {

    } else {
        br_panic("Can't be here!");
    }

    br::CollectiveContext::shutdown();
}

