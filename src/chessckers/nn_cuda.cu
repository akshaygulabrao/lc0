// CUDA (cuBLAS) NN backend — CUDA port of nn_metal (MetalTrunkV2). See nn_cuda.h.
//
// Mirrors nn.hpp::ChesskersNet::trunk_v2_batch op-for-op: the conv3x3s go through im2col +
// one cuBLAS SGEMM per layer (byte-identical column layout to cc::conv3x3_batch); GroupNorm
// (two-pass mean/var in double, matching groupnorm_), ReLU, pos-emb and residual adds are
// small custom kernels. Transformer blocks (TransformerBlock2d) are supported too — LayerNorm
// and GELU custom kernels, the qkv/out_proj/ff linears + per-head attention through cuBLAS, and
// a per-row softmax kernel, mirroring transformer_block_batch_. The value/gather heads run on
// the CPU (ChesskersNet, the parity oracle), exactly like MetalTrunkV2::eval_batch.
#include "nn_cuda.h"

#include "nn.hpp"  // ChesskersNet / WeightStore (the CPU forward + parity oracle + heads)

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc {
namespace {

constexpr int H = 10, W = 10, HW = 100;  // 10x10 board
constexpr int GROUPS = 8;                // GroupNorm groups (matches the groupnorm_ G=8 calls)
constexpr int TPB = 256;                 // threads per block (power of two for the reduction)

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

inline int ceil_div(long a, int b) { return (int)((a + b - 1) / b); }

// ---- kernels -------------------------------------------------------------

// im2col for a k3p1 conv over K boards. xin is board-major [K][Cin*HW]; col is [Cin*9, COLS]
// with COLS=K*HW and board k at column offset k*HW — byte-identical to the layout that
// cc::conv3x3_batch builds on the CPU.
__global__ void im2col_kernel(const float* __restrict__ xin, float* __restrict__ col, int K,
                              int Cin) {
    const int COLS = K * HW;
    const long total = (long)Cin * 9 * COLS;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int col_pos = idx % COLS;  // k*HW + (y*W + x)
    const int row = idx / COLS;      // ci*9 + ky*3 + kx
    const int k = col_pos / HW;
    const int pix = col_pos % HW;
    const int y = pix / W, x = pix % W;
    const int ci = row / 9;
    const int kk = row % 9;
    const int ky = kk / 3, kx = kk % 3;
    const int iy = y + ky - 1, ix = x + kx - 1;
    float v = 0.0f;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
        v = xin[(long)k * Cin * HW + (long)ci * HW + iy * W + ix];
    col[idx] = v;
}

// Scatter the GEMM output [Cout, COLS] (row-major, COLS=K*HW) back to board-major
// [K][Cout*HW] — mirrors the tail of cc::conv3x3_batch.
__global__ void scatter_kernel(const float* __restrict__ gemm, float* __restrict__ dst, int K,
                               int Cout) {
    const int COLS = K * HW;
    const long total = (long)Cout * COLS;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx / COLS;
    const int rest = idx % COLS;
    const int k = rest / HW;
    const int i = rest % HW;
    dst[(long)k * Cout * HW + (long)c * HW + i] = gemm[(long)c * COLS + (long)k * HW + i];
}

// GroupNorm (optionally followed by ReLU), board-major [K][C*HW]. One block per (board, group);
// two-pass mean/variance in double to match groupnorm_ (which accumulates in double).
__global__ void groupnorm_kernel(float* __restrict__ x, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, int K, int C, float eps,
                                 int do_relu) {
    const int cg = C / GROUPS;   // channels per group
    const int n = cg * HW;       // elements per (board, group)
    const int blk = blockIdx.x;  // = k*GROUPS + g
    const int k = blk / GROUPS, g = blk % GROUPS;
    const int c0 = g * cg;
    const long base = (long)k * C * HW + (long)c0 * HW;

    extern __shared__ double sh[];  // blockDim doubles
    __shared__ double mean, inv;

    // pass 1: mean
    double s = 0.0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) s += (double)x[base + t];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) sh[threadIdx.x] += sh[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean = sh[0] / n;
    __syncthreads();

    // pass 2: variance = mean of (x-mean)^2
    double v = 0.0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const double d = (double)x[base + t] - mean;
        v += d * d;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) sh[threadIdx.x] += sh[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv = 1.0 / sqrt(sh[0] / n + (double)eps);
    __syncthreads();

    // pass 3: normalize + affine (+ relu). gamma/beta are per global channel index.
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const int c = c0 + t / HW;
        double o = ((double)x[base + t] - mean) * inv * (double)gamma[c] + (double)beta[c];
        if (do_relu && o < 0.0) o = 0.0;
        x[base + t] = (float)o;
    }
}

__global__ void relu_kernel(float* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

// x += y (same shape), board-major [K*C*HW].
__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// x += pos-emb broadcast over the K boards. x is [K][CHW], pe is [CHW].
__global__ void add_posemb_kernel(float* __restrict__ x, const float* __restrict__ pe, int K,
                                  int CHW) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)K * CHW;
    if (i < total) x[i] += pe[i % CHW];
}

// ---- transformer kernels -------------------------------------------------
// (T == HW == 100 square-tokens; head dim hd = C / n_heads.)

// channel-major board buffer [K][C*T] -> stacked token-major t[K*T, C]. idx walks the source.
__global__ void chan_to_token_kernel(const float* __restrict__ x, float* __restrict__ t, int K,
                                     int C, int T) {
    const long total = (long)K * C * T;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int s = idx % T;
    const int c = (idx / T) % C;
    const int k = idx / ((long)C * T);
    t[(long)(k * T + s) * C + c] = x[idx];  // x[k*C*T + c*T + s]
}

// inverse: token-major t[K*T, C] -> channel-major x[K][C*T].
__global__ void token_to_chan_kernel(const float* __restrict__ t, float* __restrict__ x, int K,
                                     int C, int T) {
    const long total = (long)K * C * T;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int s = idx % T;
    const int c = (idx / T) % C;
    const int k = idx / ((long)C * T);
    x[idx] = t[(long)(k * T + s) * C + c];
}

// Per-row LayerNorm over D (double mean/var, matching layernorm_). One thread per row.
__global__ void layernorm_rows_kernel(float* __restrict__ x, int rows, int D,
                                      const float* __restrict__ g, const float* __restrict__ b,
                                      float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float* row = x + (long)i * D;
    double mean = 0.0;
    for (int j = 0; j < D; ++j) mean += row[j];
    mean /= D;
    double var = 0.0;
    for (int j = 0; j < D; ++j) {
        const double d = (double)row[j] - mean;
        var += d * d;
    }
    var /= D;
    const double inv = 1.0 / sqrt(var + (double)eps);
    for (int j = 0; j < D; ++j)
        row[j] = (float)(((double)row[j] - mean) * inv * (double)g[j] + (double)b[j]);
}

// Add bias[out] broadcast over the R rows of Y[R, out].
__global__ void add_bias_kernel(float* __restrict__ Y, const float* __restrict__ b, int R,
                                int out) {
    const long total = (long)R * out;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) Y[idx] += b[idx % out];
}

// Row softmax over `cols` (double accumulation, matching the CPU attention softmax). One
// thread per row. Reads/writes in place.
__global__ void softmax_rows_kernel(float* __restrict__ s, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float* sr = s + (long)i * cols;
    float mx = sr[0];
    for (int j = 1; j < cols; ++j)
        if (sr[j] > mx) mx = sr[j];
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        const float e = (float)exp((double)sr[j] - (double)mx);
        sr[j] = e;
        sum += e;
    }
    const float inv = (float)(1.0 / sum);
    for (int j = 0; j < cols; ++j) sr[j] *= inv;
}

// Exact erf GELU, matching gelu_.
__global__ void gelu_kernel(float* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double z = x[i];
    x[i] = (float)(0.5 * z * (1.0 + erf(z * 0.70710678118654752440)));
}

// Gather head `h` of board `k` from qkv[R, 3C] into contiguous Qh/Kh/Vh [T, hd]. PyTorch packs
// MultiheadAttention as in_proj [3C, C]: Q at [0,C), K at [C,2C), V at [2C,3C); head h is the
// hd-wide slice at offset off = h*hd within each.
__global__ void gather_qkv_kernel(const float* __restrict__ qkv, float* __restrict__ Qh,
                                  float* __restrict__ Kh, float* __restrict__ Vh, int base, int C,
                                  int hd, int off) {
    const int n = HW * hd;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int s = idx / hd, d = idx % hd;
    const float* row = qkv + (long)(base + s) * 3 * C;
    Qh[idx] = row[off + d];
    Kh[idx] = row[C + off + d];
    Vh[idx] = row[2 * C + off + d];
}

// Scatter Oh[T, hd] for head `h` of board `k` back into attn_out[R, C].
__global__ void scatter_oh_kernel(const float* __restrict__ Oh, float* __restrict__ attn, int base,
                                  int C, int hd, int off) {
    const int n = HW * hd;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int s = idx / hd, d = idx % hd;
    attn[(long)(base + s) * C + off + d] = Oh[idx];
}

// ---- head kernels (value + policy gather heads on the GPU) -----------------

// Global mean-pool F[K][C*HW] over the 100 squares -> pooled[K,C] (double accum, /HW),
// matching value_v2's pool. One thread per (board, channel).
__global__ void pool_mean_kernel(const float* __restrict__ F, float* __restrict__ pooled, int K,
                                 int C) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // k*C + c
    if (idx >= (long)K * C) return;
    const int k = (int)(idx / C), c = (int)(idx % C);
    const float* base = F + (long)k * C * HW + (long)c * HW;
    double s = 0.0;
    for (int j = 0; j < HW; ++j) s += (double)base[j];
    pooled[idx] = (float)(s / HW);
}

// Gather FF[i,c]=F[board[i]][c][from[i]] and TF[i,c]=F[...][to[i]] (policy_logits_v2 gather).
__global__ void gather_endpoints_kernel(const float* __restrict__ F, const int* __restrict__ board,
                                        const int* __restrict__ from, const int* __restrict__ to,
                                        float* __restrict__ FF, float* __restrict__ TF, int M,
                                        int C) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // i*C + c
    if (idx >= (long)M * C) return;
    const int i = (int)(idx / C), c = (int)(idx % C);
    const long fbase = (long)board[i] * C * HW + (long)c * HW;
    FF[idx] = F[fbase + from[i]];
    TF[idx] = F[fbase + to[i]];
}

// Path-mean PF[i,c] = (sum_j mask[i,j]*F[board[i]][c][j]) / denom[i] (double accum, skip 0s).
__global__ void path_mean_kernel(const float* __restrict__ F, const int* __restrict__ board,
                                 const float* __restrict__ mask, const float* __restrict__ denom,
                                 float* __restrict__ PF, int M, int C) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // i*C + c
    if (idx >= (long)M * C) return;
    const int i = (int)(idx / C), c = (int)(idx % C);
    const float* fb = F + (long)board[i] * C * HW + (long)c * HW;
    const float* mk = mask + (long)i * HW;
    double s = 0.0;
    for (int j = 0; j < HW; ++j) {
        const float pj = mk[j];
        if (pj != 0.0f) s += (double)pj * (double)fb[j];
    }
    PF[idx] = (float)(s / (double)denom[i]);
}

// Pack ctx_in[i] = [FF[i] | TF[i] | PF[i] | typ[i]] (3C+ntyp), matching policy_logits_v2.
__global__ void assemble_ctx_kernel(const float* __restrict__ FF, const float* __restrict__ TF,
                                    const float* __restrict__ PF, const float* __restrict__ typ,
                                    float* __restrict__ ctx, int M, int C, int ntyp) {
    const int cin = 3 * C + ntyp;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;  // i*cin + j
    if (idx >= (long)M * cin) return;
    const int i = (int)(idx / cin), j = (int)(idx % cin);
    float v;
    if (j < C) v = FF[(long)i * C + j];
    else if (j < 2 * C) v = TF[(long)i * C + (j - C)];
    else if (j < 3 * C) v = PF[(long)i * C + (j - 2 * C)];
    else v = typ[(long)i * ntyp + (j - 3 * C)];
    ctx[idx] = v;
}

// logit[i] = dot(src[i],tgt[i])/scale + ctx[i] (double-accumulated dot, matching policy_logits_v2).
__global__ void rowdot_kernel(const float* __restrict__ src, const float* __restrict__ tgt,
                              const float* __restrict__ ctx, float* __restrict__ logit, int M,
                              int dh, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    const float* sp = src + (long)i * dh;
    const float* tp = tgt + (long)i * dh;
    double d = 0.0;
    for (int j = 0; j < dh; ++j) d += (double)sp[j] * (double)tp[j];
    logit[i] = (float)(d / (double)scale) + ctx[i];
}

// Y[R,out] = X[R,in] @ W[out,in]^T + bias (W row-major [out,in]); mirrors linear_batch.
// Hoisted from run()'s tf lambda so the value/policy heads share it (R = K or M).
inline void lin_gemm(cublasHandle_t handle, const float* Wd, const float* bd, const float* X,
                     float* Y, int out, int in_, int R) {
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, R, in_, &one, Wd, in_, X, in_, &zero, Y, out);
    add_bias_kernel<<<ceil_div((long)R * out, TPB), TPB>>>(Y, bd, R, out);
}

// ---- device buffers ------------------------------------------------------

struct DBuf {
    float* p = nullptr;
    size_t cap = 0;
    void ensure(size_t n) {
        if (n <= cap) return;
        if (p) cudaFree(p);
        cuda_check(cudaMalloc(&p, n * sizeof(float)), "cudaMalloc scratch");
        cap = n;
    }
    ~DBuf() {
        if (p) cudaFree(p);
    }
};

// Int device buffer (grow-on-demand), for the policy head's gather indices.
struct IBuf {
    int* p = nullptr;
    size_t cap = 0;
    void ensure(size_t n) {
        if (n <= cap) return;
        if (p) cudaFree(p);
        cuda_check(cudaMalloc(&p, n * sizeof(int)), "cudaMalloc iscratch");
        cap = n;
    }
    ~IBuf() {
        if (p) cudaFree(p);
    }
};

float* upload(const std::vector<float>& v) {
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, v.size() * sizeof(float)), "cudaMalloc weight");
    cuda_check(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy weight");
    return d;
}

}  // namespace

struct CudaTrunkV2::Impl {
    const ChesskersNet* net = nullptr;
    int c_in = 16, C = 96, n_heads = 4;
    bool ok = false;
    cublasHandle_t handle = nullptr;

    // stem (device weights)
    float* stem_w = nullptr;     // [C, c_in*9]
    float* stem_gn_g = nullptr;  // [C]
    float* stem_gn_b = nullptr;  // [C]

    struct Block {
        bool posemb = false;
        bool is_tf = false;
        float* pe = nullptr;  // posemb: [C*HW]
        // residual block
        float* conv1_w = nullptr;
        float* bn1_g = nullptr;
        float* bn1_b = nullptr;
        float* conv2_w = nullptr;
        float* bn2_g = nullptr;
        float* bn2_b = nullptr;
        // transformer block (TransformerBlock2d); ff_dim = ff_mult * C
        int ff_dim = 0;
        float* norm1_g = nullptr;
        float* norm1_b = nullptr;
        float* in_proj_w = nullptr;   // [3C, C]
        float* in_proj_b = nullptr;   // [3C]
        float* out_proj_w = nullptr;  // [C, C]
        float* out_proj_b = nullptr;  // [C]
        float* norm2_g = nullptr;
        float* norm2_b = nullptr;
        float* ff0_w = nullptr;  // [ff_dim, C]
        float* ff0_b = nullptr;  // [ff_dim]
        float* ff2_w = nullptr;  // [C, ff_dim]
        float* ff2_b = nullptr;  // [C]
    };
    std::vector<Block> blocks;
    std::vector<float*> owned;  // every device weight ptr, for cleanup

    // scratch (grown on demand; the backend serializes eval through one GPU mutex)
    mutable DBuf d_in, d_col, d_gemm, d_x, d_c1, d_c2;
    // transformer-block scratch (only grown for nets that actually have tf blocks)
    mutable DBuf d_t, d_tn, d_tn2, d_qkv, d_h1, d_q, d_k, d_v, d_sc, d_oh;

    // ---- value + policy heads on the GPU (V2) ----
    bool heads_ok = false;
    int d_hidden = 256, d_move = 114;
    float* vt0_w = nullptr; float* vt0_b = nullptr;  // value_trunk.0
    float* vt1_g = nullptr; float* vt1_b = nullptr;  // value_trunk.1 (LN)
    float* vh0_w = nullptr; float* vh0_b = nullptr;  // value_head.0
    float* vh1_g = nullptr; float* vh1_b = nullptr;  // value_head.1 (LN)
    float* vh3_w = nullptr; float* vh3_b = nullptr;  // value_head.3
    float* src_w = nullptr; float* src_b = nullptr;  // src_proj
    float* tgt_w = nullptr; float* tgt_b = nullptr;  // tgt_proj
    float* cm0_w = nullptr; float* cm0_b = nullptr;  // ctx_mlp.0
    float* cm1_g = nullptr; float* cm1_b = nullptr;  // ctx_mlp.1 (LN)
    float* cm3_w = nullptr; float* cm3_b = nullptr;  // ctx_mlp.3
    mutable DBuf d_pool, d_vt, d_v1, d_wdl;
    mutable DBuf d_FF, d_TF, d_PF, d_pmask, d_denom, d_typ, d_src, d_tgt, d_ctxin, d_hid, d_ctx, d_logit;
    mutable IBuf i_from, i_to, i_board;

    float* track(float* p) {
        owned.push_back(p);
        return p;
    }
    ~Impl() {
        for (float* p : owned)
            if (p) cudaFree(p);
        if (handle) cublasDestroy(handle);
    }
};

CudaTrunkV2::CudaTrunkV2(const ChesskersNet& net) : p_(std::make_unique<Impl>()) {
    p_->net = &net;
    p_->c_in = net.c_in;
    p_->C = net.c_filters;
    p_->n_heads = net.n_heads;

    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) return;  // no GPU -> ok stays false
    if (cudaSetDevice(0) != cudaSuccess) return;
    if (cublasCreate(&p_->handle) != CUBLAS_STATUS_SUCCESS) return;

    try {
        const auto& w = net.w;
        // stem
        p_->stem_w = p_->track(upload(w.at("position_trunk.0.weight")));
        p_->stem_gn_g = p_->track(upload(w.at("position_trunk.1.weight")));
        p_->stem_gn_b = p_->track(upload(w.at("position_trunk.1.bias")));
        // blocks — walk like trunk_v2_batch: k=3.., dispatch by which keys exist.
        for (int k = 3;; ++k) {
            const std::string pfx = "position_trunk." + std::to_string(k) + ".";
            if (w.tensors.count(pfx + "pos")) {
                Impl::Block b;
                b.posemb = true;
                b.pe = p_->track(upload(w.at(pfx + "pos")));
                p_->blocks.push_back(b);
            } else if (w.tensors.count(pfx + "conv1.weight")) {
                Impl::Block b;
                b.conv1_w = p_->track(upload(w.at(pfx + "conv1.weight")));
                b.bn1_g = p_->track(upload(w.at(pfx + "bn1.weight")));
                b.bn1_b = p_->track(upload(w.at(pfx + "bn1.bias")));
                b.conv2_w = p_->track(upload(w.at(pfx + "conv2.weight")));
                b.bn2_g = p_->track(upload(w.at(pfx + "bn2.weight")));
                b.bn2_b = p_->track(upload(w.at(pfx + "bn2.bias")));
                p_->blocks.push_back(b);
            } else if (w.tensors.count(pfx + "attn.in_proj_weight")) {  // TransformerBlock2d
                Impl::Block b;
                b.is_tf = true;
                b.ff_dim = (int)w.at(pfx + "ff.0.bias").size();
                b.norm1_g = p_->track(upload(w.at(pfx + "norm1.weight")));
                b.norm1_b = p_->track(upload(w.at(pfx + "norm1.bias")));
                b.in_proj_w = p_->track(upload(w.at(pfx + "attn.in_proj_weight")));
                b.in_proj_b = p_->track(upload(w.at(pfx + "attn.in_proj_bias")));
                b.out_proj_w = p_->track(upload(w.at(pfx + "attn.out_proj.weight")));
                b.out_proj_b = p_->track(upload(w.at(pfx + "attn.out_proj.bias")));
                b.norm2_g = p_->track(upload(w.at(pfx + "norm2.weight")));
                b.norm2_b = p_->track(upload(w.at(pfx + "norm2.bias")));
                b.ff0_w = p_->track(upload(w.at(pfx + "ff.0.weight")));
                b.ff0_b = p_->track(upload(w.at(pfx + "ff.0.bias")));
                b.ff2_w = p_->track(upload(w.at(pfx + "ff.2.weight")));
                b.ff2_b = p_->track(upload(w.at(pfx + "ff.2.bias")));
                p_->blocks.push_back(b);
            } else {
                break;  // end of trunk
            }
        }
        // value + policy head weights (V2) — uploaded once; the heads then run on the GPU.
        if (net.is_v2) {
            p_->d_hidden = net.d_hidden;
            p_->d_move = net.d_move;
            p_->vt0_w = p_->track(upload(w.at("value_trunk.0.weight")));
            p_->vt0_b = p_->track(upload(w.at("value_trunk.0.bias")));
            p_->vt1_g = p_->track(upload(w.at("value_trunk.1.weight")));
            p_->vt1_b = p_->track(upload(w.at("value_trunk.1.bias")));
            p_->vh0_w = p_->track(upload(w.at("value_head.0.weight")));
            p_->vh0_b = p_->track(upload(w.at("value_head.0.bias")));
            p_->vh1_g = p_->track(upload(w.at("value_head.1.weight")));
            p_->vh1_b = p_->track(upload(w.at("value_head.1.bias")));
            p_->vh3_w = p_->track(upload(w.at("value_head.3.weight")));
            p_->vh3_b = p_->track(upload(w.at("value_head.3.bias")));
            p_->src_w = p_->track(upload(w.at("src_proj.weight")));
            p_->src_b = p_->track(upload(w.at("src_proj.bias")));
            p_->tgt_w = p_->track(upload(w.at("tgt_proj.weight")));
            p_->tgt_b = p_->track(upload(w.at("tgt_proj.bias")));
            p_->cm0_w = p_->track(upload(w.at("ctx_mlp.0.weight")));
            p_->cm0_b = p_->track(upload(w.at("ctx_mlp.0.bias")));
            p_->cm1_g = p_->track(upload(w.at("ctx_mlp.1.weight")));
            p_->cm1_b = p_->track(upload(w.at("ctx_mlp.1.bias")));
            p_->cm3_w = p_->track(upload(w.at("ctx_mlp.3.weight")));
            p_->cm3_b = p_->track(upload(w.at("ctx_mlp.3.bias")));
            // CC_CPU_HEADS=1 forces the CPU value/gather heads (the pre-GPU-heads path) — for
            // A/B benchmarking and as an escape hatch if a GPU-head issue ever surfaces.
            p_->heads_ok = (std::getenv("CC_CPU_HEADS") == nullptr);
        }
        p_->ok = true;
    } catch (const std::exception&) {
        p_->ok = false;  // any upload failure -> fall back to CPU
    }
}

CudaTrunkV2::~CudaTrunkV2() = default;

bool CudaTrunkV2::ok() const { return p_ && p_->ok; }

int CudaTrunkV2::run_device(const std::vector<std::vector<float>>& positions) const {
    const int K = (int)positions.size();
    if (!p_->ok || K == 0) return 0;

    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const int C = p_->C, Cin = p_->c_in, COLS = K * HW;
    const int Hn = p_->n_heads, hd = (Hn > 0) ? C / Hn : C, R = K * HW;
    Impl& s = *p_;
    s.d_in.ensure((size_t)K * Cin * HW);
    s.d_col.ensure((size_t)C * 9 * COLS);  // max input channels across layers is C
    s.d_gemm.ensure((size_t)C * COLS);
    s.d_x.ensure((size_t)K * C * HW);
    s.d_c1.ensure((size_t)K * C * HW);
    s.d_c2.ensure((size_t)K * C * HW);
    // transformer scratch — only allocated for nets that actually carry tf blocks.
    int max_ff = 0;
    for (const Impl::Block& b : s.blocks)
        if (b.is_tf) max_ff = std::max(max_ff, b.ff_dim);
    if (max_ff > 0) {
        s.d_t.ensure((size_t)R * C);
        s.d_tn.ensure((size_t)R * C);
        s.d_tn2.ensure((size_t)R * C);
        s.d_qkv.ensure((size_t)R * 3 * C);
        s.d_h1.ensure((size_t)R * max_ff);
        s.d_q.ensure((size_t)HW * hd);
        s.d_k.ensure((size_t)HW * hd);
        s.d_v.ensure((size_t)HW * hd);
        s.d_sc.ensure((size_t)HW * HW);
        s.d_oh.ensure((size_t)HW * hd);
    }

    // upload positions -> d_in (board-major [K][c_in*HW]; positions[k] is flat NCHW [c_in*100])
    std::vector<float> flat((size_t)K * Cin * HW);
    for (int k = 0; k < K; ++k)
        std::copy(positions[k].begin(), positions[k].end(), &flat[(size_t)k * Cin * HW]);
    cuda_check(cudaMemcpy(s.d_in.p, flat.data(), flat.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy positions");

    const float one = 1.0f, zero = 0.0f;
    const int gn_shmem = TPB * (int)sizeof(double);

    // conv k3p1: out[C,COLS] = W[C, srcCin*9] @ im2col(src) — one GEMM for all K boards.
    auto conv = [&](float* src, int srcCin, float* wdev, float* dst) {
        im2col_kernel<<<ceil_div((long)srcCin * 9 * COLS, TPB), TPB>>>(src, s.d_col.p, K, srcCin);
        // cuBLAS is column-major: this computes the row-major out[C,COLS] (see nn_cuda design).
        cublasSgemm(s.handle, CUBLAS_OP_N, CUBLAS_OP_N, COLS, C, srcCin * 9, &one, s.d_col.p, COLS,
                    wdev, srcCin * 9, &zero, s.d_gemm.p, COLS);
        scatter_kernel<<<ceil_div((long)C * COLS, TPB), TPB>>>(s.d_gemm.p, dst, K, C);
    };
    auto gnorm = [&](float* x, float* gamma, float* beta, int do_relu) {
        groupnorm_kernel<<<K * GROUPS, TPB, gn_shmem>>>(x, gamma, beta, K, C, 1e-5f, do_relu);
    };
    const long NX = (long)K * C * HW;

    // One pre-norm transformer block over the 100 square-tokens, in place on d_x. Mirrors
    // ChesskersNet::transformer_block_batch_ op-for-op: token-major transpose, LayerNorm, qkv
    // linear, per-(board,head) scaled-dot-product attention + softmax, out_proj residual, then
    // LayerNorm + GELU FFN residual. Linears go through cuBLAS; LN/softmax/GELU/transpose are the
    // custom kernels above. d_tn holds LN1 then is overwritten by the attention output (scatter
    // writes every (token,channel) exactly once), so no stale data survives.
    auto tf = [&](const Impl::Block& b) {
        const int ff = b.ff_dim;
        // Y[R,out] = X[R,in] @ W[out,in]^T + bias (W row-major [out,in]); mirrors linear_batch.
        // Row-major-via-column-major: same mapping the conv lambda uses, with W transposed.
        auto lin = [&](const float* Wd, const float* bd, const float* X, float* Y, int out, int in_) {
            lin_gemm(s.handle, Wd, bd, X, Y, out, in_, R);
        };
        // t = token-major(x); n = LayerNorm1(t)
        chan_to_token_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_x.p, s.d_t.p, K, C, HW);
        cuda_check(cudaMemcpy(s.d_tn.p, s.d_t.p, (size_t)R * C * sizeof(float),
                              cudaMemcpyDeviceToDevice), "tf LN1 copy");
        layernorm_rows_kernel<<<ceil_div(R, TPB), TPB>>>(s.d_tn.p, R, C, b.norm1_g, b.norm1_b, 1e-5f);
        // qkv[R,3C] = in_proj(n)
        lin(b.in_proj_w, b.in_proj_b, s.d_tn.p, s.d_qkv.p, 3 * C, C);
        // per-board, per-head attention -> attn_out in d_tn
        const float scale = 1.0f / std::sqrt((float)hd);
        for (int k = 0; k < K; ++k)
            for (int h = 0; h < Hn; ++h) {
                const int base = k * HW, off = h * hd;
                gather_qkv_kernel<<<ceil_div(HW * hd, TPB), TPB>>>(s.d_qkv.p, s.d_q.p, s.d_k.p,
                                                                  s.d_v.p, base, C, hd, off);
                // scores[T,T] = (Qh @ Kh^T) * scale ; row softmax ; Oh[T,hd] = scores @ Vh
                cublasSgemm(s.handle, CUBLAS_OP_T, CUBLAS_OP_N, HW, HW, hd, &scale, s.d_k.p, hd,
                            s.d_q.p, hd, &zero, s.d_sc.p, HW);
                softmax_rows_kernel<<<ceil_div(HW, TPB), TPB>>>(s.d_sc.p, HW, HW);
                cublasSgemm(s.handle, CUBLAS_OP_N, CUBLAS_OP_N, hd, HW, HW, &one, s.d_v.p, hd,
                            s.d_sc.p, HW, &zero, s.d_oh.p, hd);
                scatter_oh_kernel<<<ceil_div(HW * hd, TPB), TPB>>>(s.d_oh.p, s.d_tn.p, base, C, hd,
                                                                  off);
            }
        // t += out_proj(attn_out)
        lin(b.out_proj_w, b.out_proj_b, s.d_tn.p, s.d_tn2.p, C, C);
        add_kernel<<<ceil_div((long)R * C, TPB), TPB>>>(s.d_t.p, s.d_tn2.p, (long)R * C);
        // n2 = LayerNorm2(t) ; h1 = GELU(ff0(n2)) ; t += ff2(h1)
        cuda_check(cudaMemcpy(s.d_tn.p, s.d_t.p, (size_t)R * C * sizeof(float),
                              cudaMemcpyDeviceToDevice), "tf LN2 copy");
        layernorm_rows_kernel<<<ceil_div(R, TPB), TPB>>>(s.d_tn.p, R, C, b.norm2_g, b.norm2_b, 1e-5f);
        lin(b.ff0_w, b.ff0_b, s.d_tn.p, s.d_h1.p, ff, C);
        gelu_kernel<<<ceil_div((long)R * ff, TPB), TPB>>>(s.d_h1.p, (long)R * ff);
        lin(b.ff2_w, b.ff2_b, s.d_h1.p, s.d_tn2.p, C, ff);
        add_kernel<<<ceil_div((long)R * C, TPB), TPB>>>(s.d_t.p, s.d_tn2.p, (long)R * C);
        // x = channel-major(t)
        token_to_chan_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_t.p, s.d_x.p, K, C, HW);
    };

    // stem: conv -> groupnorm -> relu
    conv(s.d_in.p, Cin, s.stem_w, s.d_x.p);
    gnorm(s.d_x.p, s.stem_gn_g, s.stem_gn_b, /*relu=*/1);

    // blocks
    for (const Impl::Block& b : s.blocks) {
        if (b.posemb) {
            add_posemb_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_x.p, b.pe, K, C * HW);
        } else if (b.is_tf) {
            tf(b);
        } else {
            conv(s.d_x.p, C, b.conv1_w, s.d_c1.p);        // c1 = conv1(x)
            gnorm(s.d_c1.p, b.bn1_g, b.bn1_b, /*relu=*/1);
            conv(s.d_c1.p, C, b.conv2_w, s.d_c2.p);       // c2 = conv2(c1)
            gnorm(s.d_c2.p, b.bn2_g, b.bn2_b, /*relu=*/0);
            add_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_c2.p, s.d_x.p, NX);  // c2 += x
            relu_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_c2.p, NX);          // relu(c2)
            cuda_check(cudaMemcpy(s.d_x.p, s.d_c2.p, NX * sizeof(float), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy residual");  // x = c2
        }
    }

    cuda_check(cudaGetLastError(), "trunk kernel launch");
    return K;  // K feature maps left in s.d_x.p (board-major [K][C*HW])
}

// Public run(): GPU trunk, then download the K feature maps to host.
std::vector<std::vector<float>> CudaTrunkV2::run(
    const std::vector<std::vector<float>>& positions) const {
    std::vector<std::vector<float>> result;
    const int K = run_device(positions);
    if (K == 0) return result;
    const int C = p_->C;
    const long NX = (long)K * C * HW;
    std::vector<float> outf((size_t)NX);
    cuda_check(cudaMemcpy(outf.data(), p_->d_x.p, NX * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy result");
    result.resize(K);
    for (int k = 0; k < K; ++k)
        result[k].assign(&outf[(size_t)k * C * HW], &outf[(size_t)(k + 1) * C * HW]);
    return result;
}

// Run the value + policy heads on the K feature maps already in s.d_x.p (left by run_device or
// uploaded by eval_heads_from_F). Mirrors value_v2/policy_logits_v2 op-for-op; only the
// per-board softmaxes run on the host (softmax_priors). Returns K (value, priors).
std::vector<std::pair<float, std::vector<float>>> CudaTrunkV2::eval_heads_device(
    int K, const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    std::vector<std::pair<float, std::vector<float>>> out(K);
    Impl& s = *p_;
    const int C = s.C, dh = s.d_hidden;

    // value head: pool -> value_trunk.0 -> LN -> relu -> value_head.0 -> LN -> relu -> value_head.3
    s.d_pool.ensure((size_t)K * C);
    s.d_vt.ensure((size_t)K * dh);
    s.d_v1.ensure((size_t)K * (dh / 2));
    s.d_wdl.ensure((size_t)K * 3);
    pool_mean_kernel<<<ceil_div((long)K * C, TPB), TPB>>>(s.d_x.p, s.d_pool.p, K, C);
    lin_gemm(s.handle, s.vt0_w, s.vt0_b, s.d_pool.p, s.d_vt.p, dh, C, K);
    layernorm_rows_kernel<<<ceil_div(K, TPB), TPB>>>(s.d_vt.p, K, dh, s.vt1_g, s.vt1_b, 1e-5f);
    relu_kernel<<<ceil_div((long)K * dh, TPB), TPB>>>(s.d_vt.p, (long)K * dh);
    lin_gemm(s.handle, s.vh0_w, s.vh0_b, s.d_vt.p, s.d_v1.p, dh / 2, dh, K);
    layernorm_rows_kernel<<<ceil_div(K, TPB), TPB>>>(s.d_v1.p, K, dh / 2, s.vh1_g, s.vh1_b, 1e-5f);
    relu_kernel<<<ceil_div((long)K * (dh / 2), TPB), TPB>>>(s.d_v1.p, (long)K * (dh / 2));
    lin_gemm(s.handle, s.vh3_w, s.vh3_b, s.d_v1.p, s.d_wdl.p, 3, dh / 2, K);
    std::vector<float> wdl((size_t)K * 3);
    cuda_check(cudaMemcpy(wdl.data(), s.d_wdl.p, (size_t)K * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost), "wdl download");

    // policy head: flattened over the M moves of all boards
    const FlatMoves fm = flatten_moves(moves_per, s.d_move);
    const int M = fm.M, ntyp = fm.n_typ;
    std::vector<float> logits;
    if (M > 0) {
        const int cin = 3 * C + ntyp;
        s.d_FF.ensure((size_t)M * C); s.d_TF.ensure((size_t)M * C); s.d_PF.ensure((size_t)M * C);
        s.d_pmask.ensure((size_t)M * HW); s.d_denom.ensure((size_t)M); s.d_typ.ensure((size_t)M * ntyp);
        s.d_src.ensure((size_t)M * dh); s.d_tgt.ensure((size_t)M * dh);
        s.d_ctxin.ensure((size_t)M * cin); s.d_hid.ensure((size_t)M * dh);
        s.d_ctx.ensure((size_t)M); s.d_logit.ensure((size_t)M);
        s.i_from.ensure((size_t)M); s.i_to.ensure((size_t)M); s.i_board.ensure((size_t)M);
        cuda_check(cudaMemcpy(s.i_from.p, fm.from_idx.data(), (size_t)M * sizeof(int), cudaMemcpyHostToDevice), "from up");
        cuda_check(cudaMemcpy(s.i_to.p, fm.to_idx.data(), (size_t)M * sizeof(int), cudaMemcpyHostToDevice), "to up");
        cuda_check(cudaMemcpy(s.i_board.p, fm.board_of.data(), (size_t)M * sizeof(int), cudaMemcpyHostToDevice), "board up");
        cuda_check(cudaMemcpy(s.d_pmask.p, fm.pathmask.data(), (size_t)M * HW * sizeof(float), cudaMemcpyHostToDevice), "pmask up");
        cuda_check(cudaMemcpy(s.d_denom.p, fm.denom.data(), (size_t)M * sizeof(float), cudaMemcpyHostToDevice), "denom up");
        cuda_check(cudaMemcpy(s.d_typ.p, fm.typ.data(), (size_t)M * ntyp * sizeof(float), cudaMemcpyHostToDevice), "typ up");
        gather_endpoints_kernel<<<ceil_div((long)M * C, TPB), TPB>>>(s.d_x.p, s.i_board.p, s.i_from.p, s.i_to.p, s.d_FF.p, s.d_TF.p, M, C);
        path_mean_kernel<<<ceil_div((long)M * C, TPB), TPB>>>(s.d_x.p, s.i_board.p, s.d_pmask.p, s.d_denom.p, s.d_PF.p, M, C);
        lin_gemm(s.handle, s.src_w, s.src_b, s.d_FF.p, s.d_src.p, dh, C, M);
        lin_gemm(s.handle, s.tgt_w, s.tgt_b, s.d_TF.p, s.d_tgt.p, dh, C, M);
        assemble_ctx_kernel<<<ceil_div((long)M * cin, TPB), TPB>>>(s.d_FF.p, s.d_TF.p, s.d_PF.p, s.d_typ.p, s.d_ctxin.p, M, C, ntyp);
        lin_gemm(s.handle, s.cm0_w, s.cm0_b, s.d_ctxin.p, s.d_hid.p, dh, cin, M);
        layernorm_rows_kernel<<<ceil_div(M, TPB), TPB>>>(s.d_hid.p, M, dh, s.cm1_g, s.cm1_b, 1e-5f);
        relu_kernel<<<ceil_div((long)M * dh, TPB), TPB>>>(s.d_hid.p, (long)M * dh);
        lin_gemm(s.handle, s.cm3_w, s.cm3_b, s.d_hid.p, s.d_ctx.p, 1, dh, M);
        rowdot_kernel<<<ceil_div(M, TPB), TPB>>>(s.d_src.p, s.d_tgt.p, s.d_ctx.p, s.d_logit.p, M, dh,
                                                 std::sqrt((float)dh));
        logits.resize(M);
        cuda_check(cudaMemcpy(logits.data(), s.d_logit.p, (size_t)M * sizeof(float),
                              cudaMemcpyDeviceToHost), "logits download");
    }
    cuda_check(cudaGetLastError(), "head kernel launch");

    for (int k = 0; k < K; ++k) {
        const float* z = &wdl[(size_t)k * 3];
        const float mx = std::max({z[0], z[1], z[2]});
        const double e0 = std::exp(z[0] - mx), e1 = std::exp(z[1] - mx), e2 = std::exp(z[2] - mx);
        const float v = (float)((e0 - e2) / (e0 + e1 + e2));
        const int lo = fm.board_off[k], n = fm.board_off[k + 1] - lo;
        out[k] = {v, (M > 0 && n > 0) ? softmax_priors(&logits[lo], n) : std::vector<float>()};
    }
    return out;
}

// Heads on already-computed trunk features Fs (uploads F, then runs the GPU heads). For the
// head-isolation parity test: feed the SAME F here and to the CPU oracle.
std::vector<std::pair<float, std::vector<float>>> CudaTrunkV2::eval_heads_from_F(
    const std::vector<std::vector<float>>& Fs,
    const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    const int K = (int)Fs.size();
    if (!p_->ok || !p_->heads_ok || K == 0)
        return std::vector<std::pair<float, std::vector<float>>>(K);
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    Impl& s = *p_;
    const int C = s.C;
    s.d_x.ensure((size_t)K * C * HW);
    std::vector<float> flat((size_t)K * C * HW);
    for (int k = 0; k < K; ++k) std::copy(Fs[k].begin(), Fs[k].end(), &flat[(size_t)k * C * HW]);
    cuda_check(cudaMemcpy(s.d_x.p, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice),
               "F upload");
    return eval_heads_device(K, moves_per);
}

std::vector<std::pair<float, std::vector<float>>> CudaTrunkV2::eval_batch(
    const std::vector<std::vector<float>>& positions,
    const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    const int K = (int)positions.size();
    std::vector<std::pair<float, std::vector<float>>> out(K);
    if (!p_->ok || !p_->net) return out;
    if (p_->heads_ok) {
        const int kk = run_device(positions);  // GPU trunk, leaves F in d_x.p
        if (kk == 0) return out;
        return eval_heads_device(kk, moves_per);  // GPU value + policy heads
    }
    // Fallback (no V2 heads): GPU trunk + CPU heads.
    const auto Fs = run(positions);
    const ChesskersNet& net = *p_->net;
    for (int k = 0; k < K; ++k) {
        const float v = net.value_v2(Fs[k]);
        const int N = (int)moves_per[k].size();
        if (N == 0) { out[k] = {v, std::vector<float>()}; continue; }
        const auto logits = net.policy_logits_v2(Fs[k], moves_per[k]);
        out[k] = {v, softmax_priors(logits.data(), N)};
    }
    return out;
}

}  // namespace cc
