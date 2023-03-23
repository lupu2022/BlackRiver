#include <unistd.h>

#include "common.hpp"
#include "context.hpp"
#include "dag.hpp"
#include "nn.hpp"

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

    } else if ( br::CollectiveContext::mpi_rank == 1 || br::CollectiveContext::mpi_rank == 2) {
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(15 + 1);
        br::load_nn_words(*env);

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
            delete init_wd;
        }

        sleep(5);

        delete env;
    } else if ( br::CollectiveContext::mpi_rank == 3) {

    } else {
        br_panic("Can't be here!");
    }

    br::CollectiveContext::shutdown();
}

