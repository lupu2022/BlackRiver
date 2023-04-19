#include <cooperative_groups.h>
#include <cstddef>


#include "common.h"
#include "block_reduce.h"
#include "kernels.h"

#define TILE_DIM 32

namespace cg = cooperative_groups;
namespace kernels {

template <typename T>
__forceinline__ __device__ T add_eps(T x, float eps) {
  return fabsf(x) > eps ? x : (x < 0 ? -eps : eps);
}


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size* seq_len, hidden_size], ln result.
vars: [batch_size* seq_len], variance per token
means: [batch_size* seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
__global__ void ker_layer_norm(float *ln_res, float *vars, float *means, const float *inp,
                               const float *scale, const float *bias, int hidden_size, float eps) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum +=
        val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + eps;
    if (vars != nullptr) {
        vars[blockIdx.x] = s_var;
    }
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
  }
}

void launch_layer_norm_float(float *ln_res, float *vars, float *means,
                          const float *inp, const float *scale,
                          const float *bias, int batch_size, int hidden_dim, float eps, 
                          cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
      throw ::std::runtime_error("violate hidden_dim % 4(float) 8(__half) = 0");
  }
  hidden_dim = hidden_dim >> 2;

  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim, eps);
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad,
                               const T *residual_grad, const T *inp_or_out,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, const uint8_t *cmask,
                               int hidden_dim, float eps) {
  int offset = blockIdx.x * hidden_dim + threadIdx.x;
  float4 dxhat, xhat;
  float var_rsqrt;
  float temp_cmax_g;
  if (threadIdx.x < hidden_dim) {
    // step 0. dxhat = dout * gamma
    dxhat = ((const float4 *)out_grad)[offset];
    if (cmask) {
      uint32_t cmask4 = reinterpret_cast<const uint32_t *>(cmask)[offset];
      uint8_t *cm = reinterpret_cast<uint8_t *>(&cmask4);
      clip_bwd(dxhat.x, temp_cmax_g, dxhat.x, cm[0], 2);
      clip_bwd(dxhat.y, temp_cmax_g, dxhat.y, cm[1], 2);
      clip_bwd(dxhat.z, temp_cmax_g, dxhat.z, cm[2], 2);
      clip_bwd(dxhat.w, temp_cmax_g, dxhat.w, cm[3], 2);
    }
    float4 vgamma = ((const float4 *)gamma)[threadIdx.x];
    dxhat.x *= vgamma.x;
    dxhat.y *= vgamma.y;
    dxhat.z *= vgamma.z;
    dxhat.w *= vgamma.w;

    /*
    step 1. xhat = (output - betta) / gamma or
    (input - mean) * rsqrtf(var)
    */
    xhat = ((const float4 *)inp_or_out)[offset];
    var_rsqrt = rsqrtf((float)vars[blockIdx.x] + eps);
    if (means == nullptr) {
      // inp_or_out is output, xhat = (output - betta) / gamma
      float4 vbetta = ((const float4 *)betta)[threadIdx.x];
      xhat.x = (xhat.x - vbetta.x) / add_eps(vgamma.x, eps);
      xhat.y = (xhat.y - vbetta.y) / add_eps(vgamma.y, eps);
      xhat.z = (xhat.z - vbetta.z) / add_eps(vgamma.z, eps);
      xhat.w = (xhat.w - vbetta.w) / add_eps(vgamma.w, eps);
    } else {
      // inp_or_out is input, xhat = (input - mean) * rsqrtf(var)
      float fmean = (float)means[blockIdx.x];
      xhat.x = (xhat.x - fmean) * var_rsqrt;
      xhat.y = (xhat.y - fmean) * var_rsqrt;
      xhat.z = (xhat.z - fmean) * var_rsqrt;
      xhat.w = (xhat.w - fmean) * var_rsqrt;
    }
  }

  /* step2. block reduce sum for dxhat and dxhat*xhat */
  float reduce_val[2] = {0.f, 0.f};
  if (threadIdx.x < hidden_dim) {
    reduce_val[0] = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    reduce_val[1] = dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z +
                    dxhat.w * xhat.w;
  }
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    float mean_dim = hidden_dim * 4;
    s_sum_dxhat = reduce_val[0] / mean_dim;
    s_sum_dxhat_xhat = reduce_val[1] / mean_dim;
  }
  __syncthreads();

  /*
  step3. compute input gradient
  (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / mean_dim) * rsqrt(var)
  */
  if (threadIdx.x >= hidden_dim) {
    return;
  }
  dxhat.x = (dxhat.x - s_sum_dxhat - xhat.x * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.y = (dxhat.y - s_sum_dxhat - xhat.y * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.z = (dxhat.z - s_sum_dxhat - xhat.z * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.w = (dxhat.w - s_sum_dxhat - xhat.w * s_sum_dxhat_xhat) * var_rsqrt;
  if (residual_grad) {
    // Add the residual grad,
    // usually in pre-layer-norm for transformer layer
    float4 dresidual = ((const float4 *)residual_grad)[offset];
    dxhat.x += dresidual.x;
    dxhat.y += dresidual.y;
    dxhat.z += dresidual.z;
    dxhat.w += dresidual.w;
  }
  ((float4 *)inp_grad)[offset] = dxhat;
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma


@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        T *cmax_grad, const T *out_grad,
                                        const T *inp_or_out, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, const uint8_t *cmask,
                                        int rows, int width, float eps) {
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * width + idx;
  int y_stride = width * TILE_DIM;

  // Loop across inp height
  float dbetta = 0;
  float dgamma = 0;
  float dout, val;

  float thread_cmax_g = 0, cmax_g;
  if (idx < width) {
    if (means == nullptr) {
      float vbetta = (float)betta[idx];
      float vgamma = (float)gamma[idx];
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        if (cmask) {
          clip_bwd(dout, cmax_g, dout, cmask[offset], 2);
          thread_cmax_g += cmax_g;
        }
        // inp_or_out is output
        val = (float)inp_or_out[offset];
        dbetta += dout;
        dgamma += ((val - vbetta) / add_eps(vgamma, eps) * dout);
        offset += y_stride;
      }
    } else {
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        if (cmask) {
          clip_bwd(dout, cmax_g, dout, cmask[offset], 2);
          thread_cmax_g += cmax_g;
        }
        // inp_or_out is input
        val = (float)inp_or_out[offset];
        dbetta += dout;
        dgamma += ((val - (float)means[r]) *
                   rsqrtf((float)vars[r] + eps) * dout);
        offset += y_stride;
      }
    }
  }
  __shared__ float block_cmax_g;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_cmax_g = 0;

  // Sum the shared buffer.
  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  __syncthreads();

  if (thread_cmax_g != 0) {
    atomicAdd(&block_cmax_g, thread_cmax_g);
  }

  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (cmask && block_cmax_g != 0) {
      atomicAdd(&cmax_grad[0], block_cmax_g);
    }
  }

  for (int i = 1; i < TILE_DIM; i <<= 1) {
    s1 += g.shfl_down(s1, i);
    s2 += g.shfl_down(s2, i);
  }

  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  if (threadIdx.x == 0 && idx < width) {
    betta_grad[pos] = s1;
    gamma_grad[pos] = s2;
  }
}



/**
Layer norm backword,
  compute the gradient of gamma, betta and input.
dbetta = sum(dout, dim=0)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
  (output - betta) / gamma if mean is nullptr
dgamma = sum(xhat * dout, dim=0)
dxhat = dout * gamma
dinp = (dxhat - (sum(dxhat, 1) + xhat * sum(dxhat * xhat, 1)) / hidden_dim)
  * rsqrt(var)

residual_grad, means, betta can be nullptr.
residual_grad will be added to dinp if it is not nullptr
  which is useful in transformer layer when pre-ln
means and betta are only used to compute xhat,
  (means == nullptr) ^ (betta == nullptr) should be true
*/
void launch_ln_bw_float(float *gamma_grad, float *betta_grad, float *inp_grad,
                  const float *out_grad, const float *residual_grad,
                  const float *inp_or_out, const float *gamma,
                  const float *betta, const float *vars,
                  const float *means, int batch, int hidden_dim, float eps,
                  cudaStream_t stream[2]) {
  // compute grad of gamma and betta
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream[0]>>>(
      gamma_grad, betta_grad, nullptr, out_grad, inp_or_out, gamma, betta, vars,
      means, nullptr, batch, hidden_dim, eps);
#if 0
  // compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch, nthread, 0, stream[1]>>>(
      inp_grad, out_grad, residual_grad, inp_or_out, gamma, betta, vars, means,
      nullptr, hidden_dim, eps);
#endif
}


}
