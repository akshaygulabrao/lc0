// CUDA (cuBLAS) NN backend — CUDA port of nn_metal (MetalTrunkV2). See nn_cuda.h.
//
// Mirrors nn.hpp::ChesskersNet::trunk_v2_batch op-for-op: the conv3x3s go through im2col +
// one cuBLAS SGEMM per layer (byte-identical column layout to cc::conv3x3_batch); GroupNorm
// (two-pass mean/var in double, matching groupnorm_), ReLU, pos-emb and residual adds are
// small custom kernels. The value/gather heads run on the CPU (ChesskersNet, the parity
// oracle), exactly like MetalTrunkV2::eval_batch.
#include "nn_cuda.h"

#include "nn.hpp"  // ChesskersNet / WeightStore (the CPU forward + parity oracle + heads)

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
    int c_in = 16, C = 96;
    bool ok = false;
    cublasHandle_t handle = nullptr;

    // stem (device weights)
    float* stem_w = nullptr;     // [C, c_in*9]
    float* stem_gn_g = nullptr;  // [C]
    float* stem_gn_b = nullptr;  // [C]

    struct Block {
        bool posemb = false;
        float* pe = nullptr;  // posemb: [C*HW]
        float* conv1_w = nullptr;
        float* bn1_g = nullptr;
        float* bn1_b = nullptr;
        float* conv2_w = nullptr;
        float* bn2_g = nullptr;
        float* bn2_b = nullptr;
    };
    std::vector<Block> blocks;
    std::vector<float*> owned;  // every device weight ptr, for cleanup

    // scratch (grown on demand; the backend serializes eval through one GPU mutex)
    mutable DBuf d_in, d_col, d_gemm, d_x, d_c1, d_c2;

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
            } else if (w.tensors.count(pfx + "attn.in_proj_weight")) {
                // Transformer block: not supported by this CUDA MVP. Leave ok=false so the
                // chessckers backend falls back to the CPU forward (the same escape hatch as
                // Metal's "unsupported block"). Deployed nets use --tf-blocks=0 (pure ResNet).
                return;
            } else {
                break;  // end of trunk
            }
        }
        p_->ok = true;
    } catch (const std::exception&) {
        p_->ok = false;  // any upload failure -> fall back to CPU
    }
}

CudaTrunkV2::~CudaTrunkV2() = default;

bool CudaTrunkV2::ok() const { return p_ && p_->ok; }

std::vector<std::vector<float>> CudaTrunkV2::run(
    const std::vector<std::vector<float>>& positions) const {
    std::vector<std::vector<float>> result;
    const int K = (int)positions.size();
    if (!p_->ok || K == 0) return result;

    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const int C = p_->C, Cin = p_->c_in, COLS = K * HW;
    Impl& s = *p_;
    s.d_in.ensure((size_t)K * Cin * HW);
    s.d_col.ensure((size_t)C * 9 * COLS);  // max input channels across layers is C
    s.d_gemm.ensure((size_t)C * COLS);
    s.d_x.ensure((size_t)K * C * HW);
    s.d_c1.ensure((size_t)K * C * HW);
    s.d_c2.ensure((size_t)K * C * HW);

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

    // stem: conv -> groupnorm -> relu
    conv(s.d_in.p, Cin, s.stem_w, s.d_x.p);
    gnorm(s.d_x.p, s.stem_gn_g, s.stem_gn_b, /*relu=*/1);

    // blocks
    for (const Impl::Block& b : s.blocks) {
        if (b.posemb) {
            add_posemb_kernel<<<ceil_div(NX, TPB), TPB>>>(s.d_x.p, b.pe, K, C * HW);
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

    // download K feature maps [C*HW] (the blocking memcpy also waits on all prior kernels).
    std::vector<float> outf((size_t)NX);
    cuda_check(cudaMemcpy(outf.data(), s.d_x.p, NX * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy result");
    cuda_check(cudaGetLastError(), "kernel launch");
    result.resize(K);
    for (int k = 0; k < K; ++k)
        result[k].assign(&outf[(size_t)k * C * HW], &outf[(size_t)(k + 1) * C * HW]);
    return result;
}

std::vector<std::pair<float, std::vector<float>>> CudaTrunkV2::eval_batch(
    const std::vector<std::vector<float>>& positions,
    const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    const int K = (int)positions.size();
    std::vector<std::pair<float, std::vector<float>>> out(K);
    if (!p_->ok || !p_->net) return out;
    const auto Fs = run(positions);  // GPU trunk
    const ChesskersNet& net = *p_->net;
    for (int k = 0; k < K; ++k) {
        const float v = net.value_v2(Fs[k]);
        const int N = (int)moves_per[k].size();
        std::vector<float> priors(N);
        if (N == 0) {
            out[k] = {v, priors};
            continue;
        }
        const auto logits = net.policy_logits_v2(Fs[k], moves_per[k]);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (float l : logits) sum += std::exp(l - mx);
        for (int i = 0; i < N; ++i) priors[i] = (float)(std::exp(logits[i] - mx) / sum);
        out[k] = {v, priors};
    }
    return out;
}

}  // namespace cc
