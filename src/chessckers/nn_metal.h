// Apple-only Metal (MPSGraph) NN backend for the native engine — Phase 6.
//
// The CPU forward (nn.hpp, cblas_sgemm) is the portable default + the parity oracle. On Apple
// this Metal backend runs the SAME forward on the GPU with FULL leaf batches, where the win is
// large: a GIL-free full-batch MPS trunk forward measured ~10x/board vs batch-1 (the lc0 reason
// batching exists — it's a GPU play). Compiled only when CC_HAVE_METAL is defined (APPLE).
#pragma once

#include <memory>
#include <vector>

namespace cc {

struct ChesskersNet;  // nn.hpp

// 6a toolchain spike: C = A[M,K] @ B[K,N] via MPSGraph on the default Metal device; returns the
// max abs diff vs a CPU reference (≈0 ⇒ MPSGraph links + runs + agrees with BLAS). -1 if no GPU.
float metal_matmul_selftest(int M, int K, int N);

// Phase 6b: the V2 spatial trunk as an MPSGraph, run on the GPU with a FULL leaf batch. Mirrors
// nn.hpp ChesskersNet::trunk_v2 op-for-op (stem conv + posemb/residual/transformer blocks) and is
// held within ~1e-3 of the CPU trunk_v2_batch oracle. Pimpl so the Obj-C++/Metal types never leak
// into the C++ translation units. The graph is built once from the net's weights at construction.
class MetalTrunkV2 {
  public:
    explicit MetalTrunkV2(const ChesskersNet& net);
    ~MetalTrunkV2();
    MetalTrunkV2(const MetalTrunkV2&) = delete;
    MetalTrunkV2& operator=(const MetalTrunkV2&) = delete;

    bool ok() const;  // false if no Metal device or the trunk has an unsupported block

    // positions: K boards, each a flat NCHW [c_in*100]; returns K feature maps [c_filters*100].
    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& positions) const;

    // 6c: end-to-end batched eval — GPU trunk (cached graph) + the parity-locked CPU value/gather
    // heads per board. Byte-equivalent to ChesskersNet::eval_batch up to GPU-trunk float error.
    // Returns K (value, priors) pairs. Hold this object to keep the graph cached across calls.
    std::vector<std::pair<float, std::vector<float>>> eval_batch(
        const std::vector<std::vector<float>>& positions,
        const std::vector<std::vector<std::vector<float>>>& moves_per) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cc
