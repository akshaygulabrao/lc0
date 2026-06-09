// CUDA NN backend for the native engine — CUDA port of nn_metal (MetalTrunkV2).
//
// The CPU forward (nn.hpp, cblas_sgemm) is the portable default + the parity oracle. On an
// NVIDIA box this backend runs the SAME V2 spatial trunk on the GPU with FULL leaf batches:
// the conv3x3s go through im2col + a single cuBLAS SGEMM per layer (the dominant FLOPs and
// the batching win, exactly like conv3x3_batch); groupnorm/relu/posemb/residual are small
// custom kernels. Compiled only when CC_HAVE_CUDA is defined (nvcc + cublas + cudart present).
//
// Scope: pure-ResNet V2 trunks (stem + pos-emb + residual blocks) — the deployed nets
// (trainer --tf-blocks=0). If the trunk contains a transformer block, ok()==false and the
// chessckers backend falls back to the CPU forward (the same escape hatch as Metal's
// "unsupported block"). The value/gather heads always run on the CPU (Phase 6e seam).
#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace cc {

struct ChesskersNet;  // nn.hpp

// Phase 6b/CUDA: the V2 spatial trunk run on the GPU with a FULL leaf batch. Mirrors
// nn.hpp ChesskersNet::trunk_v2_batch op-for-op (stem conv + pos-emb/residual blocks) and is
// held within ~1e-3 of the CPU trunk_v2_batch oracle. Pimpl so the CUDA types never leak into
// the C++ translation units. Device weights are uploaded once from the net at construction.
class CudaTrunkV2 {
  public:
    explicit CudaTrunkV2(const ChesskersNet& net);
    ~CudaTrunkV2();
    CudaTrunkV2(const CudaTrunkV2&) = delete;
    CudaTrunkV2& operator=(const CudaTrunkV2&) = delete;

    bool ok() const;  // false if no CUDA device or the trunk has an unsupported block

    // positions: K boards, each a flat NCHW [c_in*100]; returns K feature maps [c_filters*100].
    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& positions) const;

    // End-to-end batched eval — GPU trunk + the parity-locked CPU value/gather heads per board.
    // Byte-equivalent to ChesskersNet::eval_batch up to GPU-trunk float error. Returns K
    // (value, priors) pairs. Hold this object to keep the device weights resident across calls.
    std::vector<std::pair<float, std::vector<float>>> eval_batch(
        const std::vector<std::vector<float>>& positions,
        const std::vector<std::vector<std::vector<float>>>& moves_per) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cc
