// CUDA NN backend for the native engine — CUDA port of nn_metal (MetalTrunkV2).
//
// The CPU forward (nn.hpp, cblas_sgemm) is the portable default + the parity oracle. On an
// NVIDIA box this backend runs the SAME V2 spatial trunk on the GPU with FULL leaf batches:
// the conv3x3s go through im2col + a single cuBLAS SGEMM per layer (the dominant FLOPs and
// the batching win, exactly like conv3x3_batch); groupnorm/relu/posemb/residual are small
// custom kernels. Compiled only when CC_HAVE_CUDA is defined (nvcc + cublas + cudart present).
//
// Supports V2 trunks including TransformerBlock2d (transformer blocks run on the GPU too). The
// value + policy(gather) heads also run on the GPU now: eval_batch keeps F device-resident and
// runs the heads via cuBLAS linears + gather/path-mean/dot kernels, with only the per-board
// softmaxes on the host. The CPU forward (nn.hpp) stays the parity oracle; if a block is
// unsupported, ok()==false and the chessckers backend falls back to the CPU forward.
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

    // Run only the value/policy heads on already-computed trunk features Fs (K maps of
    // [c_filters*100]); the back half of eval_batch, exposed so a parity harness can feed the
    // SAME F here and to the CPU oracle (value_v2/policy_logits_v2), isolating the head port.
    std::vector<std::pair<float, std::vector<float>>> eval_heads_from_F(
        const std::vector<std::vector<float>>& Fs,
        const std::vector<std::vector<std::vector<float>>>& moves_per) const;

  private:
    // GPU trunk for K boards, leaving the K feature maps in the device scratch; returns K (0 if
    // not ok / empty). Shared by run() (then downloads) and eval_batch (runs heads on-device).
    int run_device(const std::vector<std::vector<float>>& positions) const;
    // Value + policy heads on the K feature maps already in the device scratch.
    std::vector<std::pair<float, std::vector<float>>> eval_heads_device(
        int K, const std::vector<std::vector<std::vector<float>>>& moves_per) const;

    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cc
