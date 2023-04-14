#include <openblas/cblas.h>
#include <time.h>

#include "common.hpp"
#include "context.hpp"

namespace br {

int ComputingContext::cuda_device = -1;
cudaStream_t ComputingContext::cuda_stream = nullptr;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
cudnnHandle_t ComputingContext::cudnn_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
void* ComputingContext::local_workspace = nullptr;
size_t ComputingContext::workspace_size = 0;

void ComputingContext::boot(int cud) {
    cuda_device = cud;

    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cuda_stream) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));

    workspace_size = 1024 * 1024 * 32 * 4;
    CUDA_CHECK( cudaMalloc(&cuda_workspace, workspace_size) );
    local_workspace = malloc( workspace_size );
}

void ComputingContext::shutdown() {
    free(local_workspace);
    CUDA_CHECK( cudaFree(cuda_workspace) );
    CUDNN_CHECK( cudnnDestroy(cudnn_handle) );
    CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
}

/**************************************************************/

int      CollectiveContext::mpi_world = -1;
int      CollectiveContext::mpi_rank = -1;
int      CollectiveContext::current = -1;

ncclUniqueId    CollectiveContext::nccl_id;
ncclComm_t      CollectiveContext::nccl_comm = nullptr;
int             CollectiveContext::nccl_rank = -1;
int             CollectiveContext::nccl_world = -1;

void CollectiveContext::boot(int argc, char* argv[], int gpus) {
    current = time(nullptr);

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    br_assert( mpi_world == gpus + 1 , "current we only support n + 1 mode!");

    if ( mpi_rank == 0 ) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if ( mpi_rank >= 1 ) {
        nccl_world = gpus;
        nccl_rank = mpi_rank - 1;

        //ComputingContext::boot(nccl_rank);
        CUDA_CHECK( cudaSetDevice(nccl_rank) );

        NCCL_CHECK(ncclCommInitRank(&nccl_comm, nccl_world, nccl_id, nccl_rank));
    }
}

void CollectiveContext::shutdown() {
    if( nccl_comm != nullptr ) {
        NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }

    MPI_Finalize();
}

int CollectiveContext::now() {
    int n = time(nullptr);
    return n - current;
}

/**************************************************************/
const size_t MemoryContext::page_size = 4096;
void* MemoryContext::root = nullptr;
size_t MemoryContext::total_size = 0;
size_t MemoryContext::currentp = 0;

void MemoryContext::boot(size_t total_bytes) {
    total_size = total_bytes;
    root = malloc(total_bytes);
    currentp = 0;
}

void MemoryContext::shutdown() {
    free(root);
}

void* MemoryContext::alloc(size_t blk_size) {
    br_assert(blk_size % page_size == 0, "block size must page aligend");
    if ( blk_size + currentp > total_size ) {
        std::cout << CollectiveContext::mpi_rank << ": " <<  currentp << " " << blk_size << std::endl;
        br_panic("Can't allocate memory, out of pre-allocating");
    }

    void* ret = (unsigned char*)root + currentp;
    currentp = currentp + blk_size;
    return ret;
}


}
