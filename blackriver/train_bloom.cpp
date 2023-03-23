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

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        std::vector<float> xinput;
        br::load_data("model/xinput.msg", xinput);

        MPI_Send(xinput.data(), xinput.size(), MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    } else if ( br::CollectiveContext::mpi_rank == 1) {
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(15 + 1);
        br::load_nn_words(*env);

        // create weight/grad/var for every layer , zero layer is GPU
        {
            std::string init_code = fileToString("model/init.words");
            br::DaG* init_wd = env->build(init_code);

            env->change(0);
            env->hash().set("$DEVICE", "cuda");
            env->run(init_wd);
            env->execute("create_weight");
            env->execute("create_grad");
            env->execute("create_var");

            for (int i = 1; i <= 15; i++) {
                env->change(i);
                env->hash().set("$DEVICE", "cpu");
                env->run( init_wd );
                env->execute("create_weight");
                env->execute("create_grad");
            }

            env->change(1);
            env->execute("'h0' load_weight");
            env->execute("1 sync");
            delete init_wd;
        }

        env->change(0);
        br::tensor_t xinput = env->hash().find_tensor("xinput");
        br::tensor_t xinput_ = env->hash().find_tensor("xinput_");
        xinput_->io_mpi_recv(xinput_, 0);

        xinput->op_copy(xinput, xinput_);

        {
            std::string train_code = fileToString("model/train.words");
            env->execute(train_code);
            env->execute("op.sync");
            std::cout << "Executing forward..." << std::endl;
            env->execute("forward");
        }

        sleep(5);

        delete env;
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        sleep(5);
    } else if ( br::CollectiveContext::mpi_rank == 3) {

    } else {
        br_panic("Can't be here!");
    }

    br::CollectiveContext::shutdown();
}

