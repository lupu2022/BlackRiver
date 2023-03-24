#include "common.hpp"
#include "context.hpp"

namespace br {

int ComputingContext::cuda_device = -1;
cudaStream_t ComputingContext::cuda_stream = nullptr;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
cudnnHandle_t ComputingContext::cudnn_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
size_t ComputingContext::cuda_workspace_size = 0;

void ComputingContext::boot(int cud) {
    cuda_device = cud;

    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cuda_stream) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));

    cuda_workspace_size = 1024 * 1024 * 32;
    CUDA_CHECK( cudaMalloc(&cuda_workspace, cuda_workspace_size) );
}

void ComputingContext::shutdown() {
    CUDNN_CHECK( cudnnDestroy(cudnn_handle) );
    CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
}

int      CollectiveContext::mpi_world = -1;
int      CollectiveContext::mpi_rank = -1;

ncclUniqueId    CollectiveContext::nccl_id;
ncclComm_t      CollectiveContext::nccl_comm = nullptr;
int             CollectiveContext::nccl_rank = -1;
int             CollectiveContext::nccl_world = -1;

void CollectiveContext::boot(int argc, char* argv[], int gpus) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    br_assert( mpi_world == gpus + 2 , "current we only support n + 2 mode!");

    if ( mpi_rank == 0 ) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if ( mpi_rank >= 1 && mpi_rank < mpi_world - 1 ) {
        nccl_world = gpus;
        nccl_rank = mpi_rank - 1;

        ComputingContext::boot(nccl_rank);

        NCCL_CHECK(ncclCommInitRank(&nccl_comm, nccl_world, nccl_id, nccl_rank));
    }
}

void CollectiveContext::shutdown() {
    if( nccl_comm != nullptr ) {
        NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }

    MPI_Finalize();
}

}
