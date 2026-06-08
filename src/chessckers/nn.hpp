// Chessckers C++ engine — Slice 6/7: native NN forward (CBLAS).
//
// Hand-rolled forward of ChesskersScorer off the exported PyTorch weights
// (native_net.export_state_dict). Linear/conv go through cblas_sgemm (conv via
// im2col); the per-leaf policy head is batched over ALL the leaf's moves in single
// GEMMs. GroupNorm/LayerNorm stay as (cheap) loops, in double. Held within ~1e-4 of
// PyTorch by tests/test_cpp_nn_parity.py.
//
//   pos_emb = position_trunk(pos[14,8,8])
//   value   = wdl[0]-wdl[2],  wdl = softmax(value_head(pos_emb))
//   logits  = head(cat[pos_emb, move_encoder(M)]) over M=[N,240];  priors = softmax
#pragma once

// BLAS backend: the forward is portable cblas_sgemm. Apple ships it in Accelerate; every
// other platform provides the same CBLAS symbols via OpenBLAS / MKL / reference BLAS. So the
// engine is cross-platform (lc0-style) — only the include + link differ per platform.
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cc {

struct NTensor {
    std::vector<int> shape;
    std::vector<float> data;
};

struct WeightStore {
    std::map<std::string, NTensor> tensors;
    const std::vector<float>& at(const std::string& k) const {
        const auto it = tensors.find(k);
        if (it == tensors.end()) throw std::runtime_error("missing weight: " + k);
        return it->second.data;
    }
};

inline WeightStore load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open weights: " + path);
    auto ri = [&]() {
        int32_t v;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    WeightStore ws;
    const int n = ri();
    for (int i = 0; i < n; ++i) {
        const int nl = ri();
        std::string name(nl, '\0');
        f.read(&name[0], nl);
        const int nd = ri();
        NTensor t;
        long prod = 1;
        for (int d = 0; d < nd; ++d) {
            const int dim = ri();
            t.shape.push_back(dim);
            prod *= dim;
        }
        t.data.resize(prod);
        f.read(reinterpret_cast<char*>(t.data.data()), prod * 4);
        if (!f) throw std::runtime_error("truncated weights for " + name);
        ws.tensors[name] = std::move(t);
    }
    return ws;
}

// --- ops (8x8 board; channel-major flat layout c*64 + y*8 + x) ---

// Y[N,out] = X[N,in] @ W[out,in]^T + b   (W is PyTorch Linear weight, row-major).
inline std::vector<float> linear_batch(const std::vector<float>& X, int N,
                                       const std::vector<float>& W, const std::vector<float>& b,
                                       int out_dim, int in_dim) {
    std::vector<float> Y((size_t)N * out_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, out_dim, in_dim, 1.0f, X.data(), in_dim,
                W.data(), in_dim, 0.0f, Y.data(), out_dim);
    for (int n = 0; n < N; ++n) {
        float* yp = &Y[(size_t)n * out_dim];
        for (int o = 0; o < out_dim; ++o) yp[o] += b[o];
    }
    return Y;
}

// conv k3p1 bias=false via im2col + GEMM. in: [Cin,H,W], w: [Cout,Cin*9] -> [Cout,H,W].
// H,W default to 8 (V1's 8x8 board); V2's spatial trunk passes 10,10. With H=W=8 the
// body is bit-identical to the original (the loop bounds become runtime, the math doesn't).
inline std::vector<float> conv3x3(const std::vector<float>& in, int Cin,
                                  const std::vector<float>& w, int Cout, int H = 8, int W = 8) {
    const int HW = H * W;
    std::vector<float> col((size_t)Cin * 9 * HW, 0.0f);
    for (int ci = 0; ci < Cin; ++ci)
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                float* cp = &col[(size_t)(ci * 9 + ky * 3 + kx) * HW];
                const float* ip = &in[(size_t)ci * HW];
                for (int y = 0; y < H; ++y) {
                    const int iy = y + ky - 1;
                    for (int x = 0; x < W; ++x) {
                        const int ix = x + kx - 1;
                        cp[y * W + x] =
                            (iy >= 0 && iy < H && ix >= 0 && ix < W) ? ip[iy * W + ix] : 0.0f;
                    }
                }
            }
    std::vector<float> out((size_t)Cout * HW);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Cout, HW, Cin * 9, 1.0f, w.data(),
                Cin * 9, col.data(), HW, 0.0f, out.data(), HW);
    return out;
}

// Batched conv k3p1 over K boards, ONE GEMM. ins[k] = [Cin,H,W] -> outs[k] = [Cout,H,W].
// im2col columns of all K boards are concatenated ([Cin*9, K*HW], board k at column k*HW)
// so a single W[Cout,Cin*9] @ col GEMM does all K — the BLAS-efficiency win (per the
// microbench: ~1.74x/board at K=8 for the 96-filter residual conv). Output is scattered
// back to per-board channel-major buffers (cheap copy). Byte-equivalent to K conv3x3 calls.
inline std::vector<std::vector<float>> conv3x3_batch(const std::vector<std::vector<float>>& ins,
                                                     int Cin, const std::vector<float>& w, int Cout,
                                                     int H = 8, int W = 8) {
    const int HW = H * W, K = (int)ins.size(), COLS = HW * K;
    std::vector<float> col((size_t)Cin * 9 * COLS, 0.0f);
    for (int k = 0; k < K; ++k)
        for (int ci = 0; ci < Cin; ++ci)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    float* cp = &col[(size_t)(ci * 9 + ky * 3 + kx) * COLS + (size_t)k * HW];
                    const float* ip = &ins[k][(size_t)ci * HW];
                    for (int y = 0; y < H; ++y) {
                        const int iy = y + ky - 1;
                        for (int x = 0; x < W; ++x) {
                            const int ix = x + kx - 1;
                            cp[y * W + x] =
                                (iy >= 0 && iy < H && ix >= 0 && ix < W) ? ip[iy * W + ix] : 0.0f;
                        }
                    }
                }
    std::vector<float> out((size_t)Cout * COLS);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Cout, COLS, Cin * 9, 1.0f, w.data(),
                Cin * 9, col.data(), COLS, 0.0f, out.data(), COLS);
    std::vector<std::vector<float>> outs(K, std::vector<float>((size_t)Cout * HW));
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < Cout; ++c) {
            const float* src = &out[(size_t)c * COLS + (size_t)k * HW];
            std::copy(src, src + HW, &outs[k][(size_t)c * HW]);
        }
    return outs;
}

inline void groupnorm_(std::vector<float>& x, int C, int G, const std::vector<float>& w,
                       const std::vector<float>& b, float eps = 1e-5f, int HW = 64) {
    const int cg = C / G;
    for (int g = 0; g < G; ++g) {
        const int c0 = g * cg, c1 = (g + 1) * cg, cnt = cg * HW;
        double mean = 0.0;
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i) mean += x[c * HW + i];
        mean /= cnt;
        double var = 0.0;
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i) {
                const double d = x[c * HW + i] - mean;
                var += d * d;
            }
        var /= cnt;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int c = c0; c < c1; ++c)
            for (int i = 0; i < HW; ++i)
                x[c * HW + i] = static_cast<float>((x[c * HW + i] - mean) * inv * w[c] + b[c]);
    }
}

// per-row LayerNorm over [N, D].
inline void layernorm_(std::vector<float>& x, int N, int D, const std::vector<float>& w,
                       const std::vector<float>& b, float eps = 1e-5f) {
    for (int n = 0; n < N; ++n) {
        float* row = &x[(size_t)n * D];
        double mean = 0.0;
        for (int i = 0; i < D; ++i) mean += row[i];
        mean /= D;
        double var = 0.0;
        for (int i = 0; i < D; ++i) {
            const double d = row[i] - mean;
            var += d * d;
        }
        var /= D;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) row[i] = static_cast<float>((row[i] - mean) * inv * w[i] + b[i]);
    }
}

inline void relu_(std::vector<float>& x) {
    for (float& v : x)
        if (v < 0) v = 0;
}

// Exact GELU (erf form) — matches PyTorch nn.GELU() default (approximate='none').
inline void gelu_(std::vector<float>& x) {
    for (float& v : x) {
        const double z = v;
        v = static_cast<float>(0.5 * z * (1.0 + std::erf(z * 0.70710678118654752440)));
    }
}

struct ChesskersNet {
    WeightStore w;
    int c_in = 15, c_filters = 96, d_hidden = 256, d_move = 240;
    bool is_v2 = false;   // ChesskersScorerV2: square-grounded gather head + (optional) transformer trunk
    int n_heads = 4;

    explicit ChesskersNet(const std::string& path) : w(load_weights(path)) {
        // V2 is detected by the gather head's source projection. It uses a 16-channel
        // 10x10 encoding + 114-dim move features (vs V1's 15ch / 8x8 / 240). d_hidden and
        // c_filters are read off src_proj.weight [d_hidden, c_filters] so the same loader
        // serves any width.
        const auto sp = w.tensors.find("src_proj.weight");
        if (sp != w.tensors.end()) {
            is_v2 = true;
            c_in = 16;
            d_move = 114;
            d_hidden = sp->second.shape[0];
            c_filters = sp->second.shape[1];
        }
    }

    std::vector<float> trunk(const std::vector<float>& pos) const {
        auto x = conv3x3(pos, c_in, w.at("position_trunk.0.weight"), c_filters);
        groupnorm_(x, c_filters, 8, w.at("position_trunk.1.weight"), w.at("position_trunk.1.bias"));
        relu_(x);
        for (int blk : {3, 4, 5, 6}) {
            const std::string p = "position_trunk." + std::to_string(blk) + ".";
            auto c1 = conv3x3(x, c_filters, w.at(p + "conv1.weight"), c_filters);
            groupnorm_(c1, c_filters, 8, w.at(p + "bn1.weight"), w.at(p + "bn1.bias"));
            relu_(c1);
            auto c2 = conv3x3(c1, c_filters, w.at(p + "conv2.weight"), c_filters);
            groupnorm_(c2, c_filters, 8, w.at(p + "bn2.weight"), w.at(p + "bn2.bias"));
            for (size_t i = 0; i < x.size(); ++i) c2[i] += x[i];
            relu_(c2);
            x = std::move(c2);
        }
        auto e = linear_batch(x, 1, w.at("position_trunk.8.weight"), w.at("position_trunk.8.bias"),
                              d_hidden, c_filters * 64);
        layernorm_(e, 1, d_hidden, w.at("position_trunk.9.weight"), w.at("position_trunk.9.bias"));
        relu_(e);
        return e;  // pos_emb [d_hidden]
    }

    float value(const std::vector<float>& pos_emb) const {
        auto v = linear_batch(pos_emb, 1, w.at("value_head.0.weight"), w.at("value_head.0.bias"),
                              d_hidden / 2, d_hidden);
        layernorm_(v, 1, d_hidden / 2, w.at("value_head.1.weight"), w.at("value_head.1.bias"));
        relu_(v);
        const auto wdl = linear_batch(v, 1, w.at("value_head.3.weight"), w.at("value_head.3.bias"),
                                      3, d_hidden / 2);
        const float mx = std::max({wdl[0], wdl[1], wdl[2]});
        const double e0 = std::exp(wdl[0] - mx), e1 = std::exp(wdl[1] - mx), e2 = std::exp(wdl[2] - mx);
        return static_cast<float>((e0 - e2) / (e0 + e1 + e2));
    }

    // Policy logits for N moves (features stacked into M[N, d_move]), batched.
    std::vector<float> policy_logits(const std::vector<float>& pos_emb,
                                     const std::vector<float>& M, int N) const {
        auto me = linear_batch(M, N, w.at("move_encoder.0.weight"), w.at("move_encoder.0.bias"),
                               d_hidden, d_move);
        layernorm_(me, N, d_hidden, w.at("move_encoder.1.weight"), w.at("move_encoder.1.bias"));
        relu_(me);
        std::vector<float> comb((size_t)N * 2 * d_hidden);
        for (int i = 0; i < N; ++i) {
            float* r = &comb[(size_t)i * 2 * d_hidden];
            std::copy(pos_emb.begin(), pos_emb.end(), r);
            std::copy(&me[(size_t)i * d_hidden], &me[(size_t)(i + 1) * d_hidden], r + d_hidden);
        }
        auto h = linear_batch(comb, N, w.at("head.0.weight"), w.at("head.0.bias"), d_hidden,
                              2 * d_hidden);
        layernorm_(h, N, d_hidden, w.at("head.1.weight"), w.at("head.1.bias"));
        relu_(h);
        return linear_batch(h, N, w.at("head.3.weight"), w.at("head.3.bias"), 1, d_hidden);  // [N]
    }

    // ===== V2 (ChesskersScorerV2): square-grounded gather head + transformer trunk =====

    // One pre-norm Transformer encoder block over the 100 square-tokens of the
    // (c_filters,10,10) map, in place on the channel-major buffer x[c_filters*100].
    // attn(LN(t)) residual, then ff(LN(t)) residual — mirrors model.TransformerBlock2d.
    void transformer_block_(std::vector<float>& x, const std::string& p) const {
        const int T = 100, C = c_filters, Hn = n_heads, hd = C / Hn;
        std::vector<float> t((size_t)T * C);                    // channel-major [C,T] -> token-major [T,C]
        for (int c = 0; c < C; ++c)
            for (int s = 0; s < T; ++s) t[(size_t)s * C + c] = x[(size_t)c * T + s];

        // ---- self-attention (pre-norm) ----
        std::vector<float> n = t;
        layernorm_(n, T, C, w.at(p + "norm1.weight"), w.at(p + "norm1.bias"));
        // PyTorch MultiheadAttention packs Q,K,V into in_proj_weight [3C,C] / in_proj_bias [3C].
        const auto qkv = linear_batch(n, T, w.at(p + "attn.in_proj_weight"),
                                      w.at(p + "attn.in_proj_bias"), 3 * C, C);
        const float scale = 1.0f / std::sqrt((float)hd);
        std::vector<float> attn_out((size_t)T * C);
        std::vector<float> Qh((size_t)T * hd), Kh((size_t)T * hd), Vh((size_t)T * hd),
            scores((size_t)T * T), Oh((size_t)T * hd);
        for (int h = 0; h < Hn; ++h) {
            const int off = h * hd;
            for (int s = 0; s < T; ++s) {
                const float* row = &qkv[(size_t)s * 3 * C];
                for (int d = 0; d < hd; ++d) {
                    Qh[(size_t)s * hd + d] = row[off + d];
                    Kh[(size_t)s * hd + d] = row[C + off + d];
                    Vh[(size_t)s * hd + d] = row[2 * C + off + d];
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, hd, scale, Qh.data(), hd,
                        Kh.data(), hd, 0.0f, scores.data(), T);            // Qh @ Kh^T / sqrt(hd)
            for (int i = 0; i < T; ++i) {                                  // row softmax (double)
                float* sr = &scores[(size_t)i * T];
                float mx = sr[0];
                for (int j = 1; j < T; ++j) mx = std::max(mx, sr[j]);
                double sum = 0.0;
                for (int j = 0; j < T; ++j) { sr[j] = (float)std::exp((double)sr[j] - mx); sum += sr[j]; }
                const float inv = (float)(1.0 / sum);
                for (int j = 0; j < T; ++j) sr[j] *= inv;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, hd, T, 1.0f, scores.data(), T,
                        Vh.data(), hd, 0.0f, Oh.data(), hd);               // scores @ Vh
            for (int s = 0; s < T; ++s)
                for (int d = 0; d < hd; ++d) attn_out[(size_t)s * C + off + d] = Oh[(size_t)s * hd + d];
        }
        const auto proj = linear_batch(attn_out, T, w.at(p + "attn.out_proj.weight"),
                                       w.at(p + "attn.out_proj.bias"), C, C);
        for (size_t i = 0; i < t.size(); ++i) t[i] += proj[i];            // residual

        // ---- feed-forward (pre-norm): Linear -> GELU -> Linear ----
        std::vector<float> n2 = t;
        layernorm_(n2, T, C, w.at(p + "norm2.weight"), w.at(p + "norm2.bias"));
        const int ff_dim = (int)w.at(p + "ff.0.bias").size();             // ff_mult * C
        auto h1 = linear_batch(n2, T, w.at(p + "ff.0.weight"), w.at(p + "ff.0.bias"), ff_dim, C);
        gelu_(h1);
        const auto h2 = linear_batch(h1, T, w.at(p + "ff.2.weight"), w.at(p + "ff.2.bias"), C, ff_dim);
        for (size_t i = 0; i < t.size(); ++i) t[i] += h2[i];              // residual

        for (int c = 0; c < C; ++c)                                       // token-major -> channel-major
            for (int s = 0; s < T; ++s) x[(size_t)c * T + s] = t[(size_t)s * C + c];
    }

    // Batched transformer block over K boards. Byte-equivalent to K transformer_block_ calls:
    // the linears (qkv/out_proj/ff0/ff2) run as ONE K*T-row GEMM each (the Accelerate-batching
    // win); LayerNorm is per-row (identical); attention stays per-board (block-diagonal, fuses
    // nothing). xs[k] = channel-major [C*100], updated in place.
    void transformer_block_batch_(std::vector<std::vector<float>>& xs, const std::string& p) const {
        const int T = 100, C = c_filters, Hn = n_heads, hd = C / Hn, K = (int)xs.size(), R = T * K;
        std::vector<float> t((size_t)R * C);                  // stacked token-major [K*T, C]
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < C; ++c)
                for (int s = 0; s < T; ++s) t[(size_t)(k * T + s) * C + c] = xs[k][(size_t)c * T + s];

        std::vector<float> n = t;
        layernorm_(n, R, C, w.at(p + "norm1.weight"), w.at(p + "norm1.bias"));
        const auto qkv = linear_batch(n, R, w.at(p + "attn.in_proj_weight"),
                                      w.at(p + "attn.in_proj_bias"), 3 * C, C);
        const float scale = 1.0f / std::sqrt((float)hd);
        std::vector<float> attn_out((size_t)R * C);
        std::vector<float> Qh((size_t)T * hd), Kh((size_t)T * hd), Vh((size_t)T * hd),
            scores((size_t)T * T), Oh((size_t)T * hd);
        for (int k = 0; k < K; ++k) {
            const size_t base = (size_t)k * T;                // first row of board k
            for (int h = 0; h < Hn; ++h) {
                const int off = h * hd;
                for (int s = 0; s < T; ++s) {
                    const float* row = &qkv[(base + s) * 3 * C];
                    for (int d = 0; d < hd; ++d) {
                        Qh[(size_t)s * hd + d] = row[off + d];
                        Kh[(size_t)s * hd + d] = row[C + off + d];
                        Vh[(size_t)s * hd + d] = row[2 * C + off + d];
                    }
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, hd, scale, Qh.data(), hd,
                            Kh.data(), hd, 0.0f, scores.data(), T);
                for (int i = 0; i < T; ++i) {
                    float* sr = &scores[(size_t)i * T];
                    float mx = sr[0];
                    for (int j = 1; j < T; ++j) mx = std::max(mx, sr[j]);
                    double sum = 0.0;
                    for (int j = 0; j < T; ++j) { sr[j] = (float)std::exp((double)sr[j] - mx); sum += sr[j]; }
                    const float inv = (float)(1.0 / sum);
                    for (int j = 0; j < T; ++j) sr[j] *= inv;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, hd, T, 1.0f, scores.data(), T,
                            Vh.data(), hd, 0.0f, Oh.data(), hd);
                for (int s = 0; s < T; ++s)
                    for (int d = 0; d < hd; ++d)
                        attn_out[(base + s) * C + off + d] = Oh[(size_t)s * hd + d];
            }
        }
        const auto proj = linear_batch(attn_out, R, w.at(p + "attn.out_proj.weight"),
                                       w.at(p + "attn.out_proj.bias"), C, C);
        for (size_t i = 0; i < t.size(); ++i) t[i] += proj[i];

        std::vector<float> n2 = t;
        layernorm_(n2, R, C, w.at(p + "norm2.weight"), w.at(p + "norm2.bias"));
        const int ff_dim = (int)w.at(p + "ff.0.bias").size();
        auto h1 = linear_batch(n2, R, w.at(p + "ff.0.weight"), w.at(p + "ff.0.bias"), ff_dim, C);
        gelu_(h1);
        const auto h2 = linear_batch(h1, R, w.at(p + "ff.2.weight"), w.at(p + "ff.2.bias"), C, ff_dim);
        for (size_t i = 0; i < t.size(); ++i) t[i] += h2[i];

        for (int k = 0; k < K; ++k)
            for (int c = 0; c < C; ++c)
                for (int s = 0; s < T; ++s) xs[k][(size_t)c * T + s] = t[(size_t)(k * T + s) * C + c];
    }

    // Spatial trunk: stem conv + (pos-emb, residual/transformer blocks). Returns the
    // (c_filters,10,10) feature map flat [c_filters*100] — NO pool/flatten (the gather
    // head needs it spatial). Blocks are walked by index and dispatched on which keys
    // exist, so any residual-first interleave (or pure ResNet) works without hardcoding.
    std::vector<float> trunk_v2(const std::vector<float>& pos) const {
        const int HW = 100;
        auto x = conv3x3(pos, c_in, w.at("position_trunk.0.weight"), c_filters, 10, 10);
        groupnorm_(x, c_filters, 8, w.at("position_trunk.1.weight"), w.at("position_trunk.1.bias"),
                   1e-5f, HW);
        relu_(x);
        for (int k = 3;; ++k) {
            const std::string p = "position_trunk." + std::to_string(k) + ".";
            if (w.tensors.count(p + "pos")) {                             // _AddSpatialPosEmb
                const auto& pe = w.at(p + "pos");                         // [1,c_filters,10,10]
                for (size_t i = 0; i < x.size(); ++i) x[i] += pe[i];
            } else if (w.tensors.count(p + "conv1.weight")) {             // ResidualBlock
                auto c1 = conv3x3(x, c_filters, w.at(p + "conv1.weight"), c_filters, 10, 10);
                groupnorm_(c1, c_filters, 8, w.at(p + "bn1.weight"), w.at(p + "bn1.bias"), 1e-5f, HW);
                relu_(c1);
                auto c2 = conv3x3(c1, c_filters, w.at(p + "conv2.weight"), c_filters, 10, 10);
                groupnorm_(c2, c_filters, 8, w.at(p + "bn2.weight"), w.at(p + "bn2.bias"), 1e-5f, HW);
                for (size_t i = 0; i < x.size(); ++i) c2[i] += x[i];
                relu_(c2);
                x = std::move(c2);
            } else if (w.tensors.count(p + "attn.in_proj_weight")) {      // TransformerBlock2d
                transformer_block_(x, p);
            } else {
                break;                                                    // end of trunk
            }
        }
        return x;
    }

    // Batched V2 spatial trunk over K boards. Mirrors trunk_v2 op-for-op but routes the
    // conv GEMMs through conv3x3_batch (one GEMM for all K — the dominant FLOPs and the
    // batching win); the cheap per-board ops (groupnorm/relu/posemb/transformer/attention)
    // stay looped. Byte-equivalent to K trunk_v2 calls. Returns K feature maps [c_filters*100].
    std::vector<std::vector<float>> trunk_v2_batch(
        const std::vector<std::vector<float>>& positions) const {
        const int HW = 100, K = (int)positions.size();
        auto xs = conv3x3_batch(positions, c_in, w.at("position_trunk.0.weight"), c_filters, 10, 10);
        for (auto& x : xs) {
            groupnorm_(x, c_filters, 8, w.at("position_trunk.1.weight"),
                       w.at("position_trunk.1.bias"), 1e-5f, HW);
            relu_(x);
        }
        for (int k = 3;; ++k) {
            const std::string p = "position_trunk." + std::to_string(k) + ".";
            if (w.tensors.count(p + "pos")) {
                const auto& pe = w.at(p + "pos");
                for (auto& x : xs)
                    for (size_t i = 0; i < x.size(); ++i) x[i] += pe[i];
            } else if (w.tensors.count(p + "conv1.weight")) {
                auto c1s = conv3x3_batch(xs, c_filters, w.at(p + "conv1.weight"), c_filters, 10, 10);
                for (auto& c1 : c1s) {
                    groupnorm_(c1, c_filters, 8, w.at(p + "bn1.weight"), w.at(p + "bn1.bias"), 1e-5f, HW);
                    relu_(c1);
                }
                auto c2s = conv3x3_batch(c1s, c_filters, w.at(p + "conv2.weight"), c_filters, 10, 10);
                for (int k2 = 0; k2 < K; ++k2) {
                    auto& c2 = c2s[k2];
                    groupnorm_(c2, c_filters, 8, w.at(p + "bn2.weight"), w.at(p + "bn2.bias"), 1e-5f, HW);
                    for (size_t i = 0; i < c2.size(); ++i) c2[i] += xs[k2][i];
                    relu_(c2);
                }
                xs = std::move(c2s);
            } else if (w.tensors.count(p + "attn.in_proj_weight")) {
                transformer_block_batch_(xs, p);
            } else {
                break;
            }
        }
        return xs;
    }

    // WDL value off the spatial map: global-mean-pool -> value_trunk -> value_head.
    float value_v2(const std::vector<float>& F) const {
        std::vector<float> pooled(c_filters);
        for (int c = 0; c < c_filters; ++c) {
            double s = 0.0;
            for (int i = 0; i < 100; ++i) s += F[(size_t)c * 100 + i];
            pooled[c] = (float)(s / 100.0);
        }
        auto vt = linear_batch(pooled, 1, w.at("value_trunk.0.weight"), w.at("value_trunk.0.bias"),
                               d_hidden, c_filters);
        layernorm_(vt, 1, d_hidden, w.at("value_trunk.1.weight"), w.at("value_trunk.1.bias"));
        relu_(vt);
        auto v = linear_batch(vt, 1, w.at("value_head.0.weight"), w.at("value_head.0.bias"),
                              d_hidden / 2, d_hidden);
        layernorm_(v, 1, d_hidden / 2, w.at("value_head.1.weight"), w.at("value_head.1.bias"));
        relu_(v);
        const auto wdl = linear_batch(v, 1, w.at("value_head.3.weight"), w.at("value_head.3.bias"),
                                      3, d_hidden / 2);
        const float mx = std::max({wdl[0], wdl[1], wdl[2]});
        const double e0 = std::exp(wdl[0] - mx), e1 = std::exp(wdl[1] - mx), e2 = std::exp(wdl[2] - mx);
        return (float)((e0 - e2) / (e0 + e1 + e2));
    }

    // Gather policy head: per move, gather F at from/to/path squares, then
    //   logit = (src_proj(F[from]) . tgt_proj(F[to])) / sqrt(d_hidden)
    //         + ctx_mlp([F[from], F[to], pathmean(F), type_scalars]).
    // Move features: [0]=from_idx, [1]=to_idx, [2:102]=path mask (100), [102:]=K scalars.
    std::vector<float> policy_logits_v2(const std::vector<float>& F,
                                        const std::vector<std::vector<float>>& moves) const {
        const int N = (int)moves.size(), C = c_filters;
        std::vector<float> FF((size_t)N * C), TF((size_t)N * C), PF((size_t)N * C);
        for (int i = 0; i < N; ++i) {
            const auto& m = moves[i];
            const int fi = (int)std::lround(m[0]), ti = (int)std::lround(m[1]);
            for (int c = 0; c < C; ++c) {
                FF[(size_t)i * C + c] = F[(size_t)c * 100 + fi];
                TF[(size_t)i * C + c] = F[(size_t)c * 100 + ti];
            }
            double denom = 0.0;
            for (int j = 0; j < 100; ++j) denom += m[2 + j];
            if (denom < 1.0) denom = 1.0;
            for (int c = 0; c < C; ++c) {
                double s = 0.0;
                for (int j = 0; j < 100; ++j) {
                    const float pj = m[2 + j];
                    if (pj != 0.0f) s += (double)pj * F[(size_t)c * 100 + j];
                }
                PF[(size_t)i * C + c] = (float)(s / denom);
            }
        }
        const auto src = linear_batch(FF, N, w.at("src_proj.weight"), w.at("src_proj.bias"), d_hidden, C);
        const auto tgt = linear_batch(TF, N, w.at("tgt_proj.weight"), w.at("tgt_proj.bias"), d_hidden, C);
        const float scale = std::sqrt((float)d_hidden);
        const int K = d_move - 102, ctx_in = 3 * C + K;                   // K type scalars (=12)
        std::vector<float> ci((size_t)N * ctx_in);
        for (int i = 0; i < N; ++i) {
            float* r = &ci[(size_t)i * ctx_in];
            for (int c = 0; c < C; ++c) r[c] = FF[(size_t)i * C + c];
            for (int c = 0; c < C; ++c) r[C + c] = TF[(size_t)i * C + c];
            for (int c = 0; c < C; ++c) r[2 * C + c] = PF[(size_t)i * C + c];
            for (int k = 0; k < K; ++k) r[3 * C + k] = moves[i][102 + k];
        }
        auto hh = linear_batch(ci, N, w.at("ctx_mlp.0.weight"), w.at("ctx_mlp.0.bias"), d_hidden, ctx_in);
        layernorm_(hh, N, d_hidden, w.at("ctx_mlp.1.weight"), w.at("ctx_mlp.1.bias"));
        relu_(hh);
        const auto ctx = linear_batch(hh, N, w.at("ctx_mlp.3.weight"), w.at("ctx_mlp.3.bias"), 1, d_hidden);
        std::vector<float> logits(N);
        for (int i = 0; i < N; ++i) {
            double d = 0.0;
            for (int j = 0; j < d_hidden; ++j)
                d += (double)src[(size_t)i * d_hidden + j] * tgt[(size_t)i * d_hidden + j];
            logits[i] = (float)(d / scale) + ctx[i];
        }
        return logits;
    }

    // V2 head: (value, softmax priors) from already-computed trunk features F. Split out of
    // eval/eval_batch so the batched driver can run the GPU/BLAS trunk ONCE for K leaves and
    // then compute these per-board heads on the game threads (Phase 6e — the heads don't batch
    // well and parallelize better off the gatherer). Byte-identical to eval's V2 tail.
    std::pair<float, std::vector<float>> head_v2(const std::vector<float>& F,
                                                 const std::vector<std::vector<float>>& moves) const {
        const float v = value_v2(F);
        const int N = (int)moves.size();
        std::vector<float> priors(N);
        if (N == 0) return {v, priors};
        const auto logits = policy_logits_v2(F, moves);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (float l : logits) s += std::exp(l - mx);
        for (int i = 0; i < N; ++i) priors[i] = (float)(std::exp(logits[i] - mx) / s);
        return {v, priors};
    }

    std::pair<float, std::vector<float>> eval(const std::vector<float>& pos,
                                              const std::vector<std::vector<float>>& moves) const {
        if (is_v2) return head_v2(trunk_v2(pos), moves);
        const auto pe = trunk(pos);
        const float v = value(pe);
        const int N = (int)moves.size();
        std::vector<float> priors(N);
        if (N == 0) return {v, priors};
        std::vector<float> M((size_t)N * d_move);
        for (int i = 0; i < N; ++i)
            std::copy(moves[i].begin(), moves[i].end(), &M[(size_t)i * d_move]);
        const auto logits = policy_logits(pe, M, N);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (float l : logits) s += std::exp(l - mx);
        for (int i = 0; i < N; ++i) priors[i] = static_cast<float>(std::exp(logits[i] - mx) / s);
        return {v, priors};
    }

    // Batched eval over K leaves: one batched trunk pass (the conv stack is fused into single
    // GEMMs), then per-board value + policy. Byte-equivalent to K eval() calls. V2 only — V1
    // falls back to a serial loop (production is V2; not worth the extra batched-trunk path).
    std::vector<std::pair<float, std::vector<float>>> eval_batch(
        const std::vector<std::vector<float>>& positions,
        const std::vector<std::vector<std::vector<float>>>& moves_per) const {
        const int K = (int)positions.size();
        std::vector<std::pair<float, std::vector<float>>> out(K);
        if (!is_v2) {
            for (int k = 0; k < K; ++k) out[k] = eval(positions[k], moves_per[k]);
            return out;
        }
        const auto Fs = trunk_v2_batch(positions);
        for (int k = 0; k < K; ++k) out[k] = head_v2(Fs[k], moves_per[k]);
        return out;
    }
};

}  // namespace cc
