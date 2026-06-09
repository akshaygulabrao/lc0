/*
  This file is part of akshay-chessckers-0 (Chessckers port of Leela Chess Zero).

  The Chessckers NN backend: an lc0 Backend that wraps the reference engine's
  cc::ChesskersNet. It replaces lc0's fixed-1858 conv-policy backends (encoder.cc/
  decoder.cc/wrapper.cc/network_legacy + the per-vendor NN backends), which assume a
  fixed policy vector and chess input planes.

  Contract (neural/backend.h): given EvalPosition{pos history, legal_moves},
  fill EvalResult{q, d, m, p[]} where p is per-legal-move (variable length), aligned
  to legal_moves by index. We:
    - encode the current board with cc::encode_position[_v2] (16ch/10x10 for V2),
    - encode each legal move's features with cc::encode_native_move,
    - run a BATCHED forward (cc::ChesskersNet::eval_batch on CPU BLAS, or
      cc::MetalTrunkV2::eval_batch on the GPU when built/available): one fused trunk
      pass over the whole minibatch + per-board value/policy heads,
    - write q=value (side-to-move WDL scalar, [-1,1]) and p[i]=priors[i].
  has_wdl/has_mlh are false for first parity (q only; no separate draw/MLH heads).

  Batching is the lc0 GPU play: the search gathers a minibatch (recommended_batch_
  size below), AddInput is called per leaf, then ComputeBlocking() runs ONE batched
  forward. On Apple with CC_HAVE_METAL the trunk runs on the GPU (~10x/board).
*/

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "chessckers/encode.hpp"
#include "chessckers/native_move.hpp"
#include "chessckers/nn.hpp"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/logging.h"

#ifdef CC_HAVE_METAL
#include "chessckers/nn_metal.h"
#endif
#ifdef CC_HAVE_CUDA
#include "chessckers/nn_cuda.h"
#endif

namespace lczero {
namespace {

using EvalBatchResult = std::vector<std::pair<float, std::vector<float>>>;

class ChesskersBackend : public Backend {
 public:
  explicit ChesskersBackend(const OptionsDict& options)
      : net_(options.Get<std::string>(SharedBackendParams::kWeightsId)) {
#ifdef CC_HAVE_METAL
    if (net_.is_v2) {
      auto metal = std::make_unique<cc::MetalTrunkV2>(net_);
      if (metal->ok()) {
        metal_ = std::move(metal);
        CERR << "Chessckers backend: Metal GPU trunk enabled (is_v2="
             << net_.is_v2 << ").";
      }
    }
#endif
#ifdef CC_HAVE_CUDA
    if (!metal_enabled() && net_.is_v2) {
      auto cuda = std::make_unique<cc::CudaTrunkV2>(net_);
      if (cuda->ok()) {
        cuda_ = std::move(cuda);
        CERR << "Chessckers backend: CUDA GPU trunk enabled (is_v2="
             << net_.is_v2 << ").";
      }
    }
#endif
    if (!gpu_enabled()) {
      CERR << "Chessckers backend: CPU BLAS forward (is_v2=" << net_.is_v2
           << ").";
    }
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = false,
        .has_wdl = false,
        .runs_on_cpu = !gpu_enabled(),
        .suggested_num_search_threads = 2,
        // Encourage the search to gather a real minibatch so the batched/GPU
        // trunk pays off (the conv GEMMs fuse across the batch).
        .recommended_batch_size = 256,
        .maximum_batch_size = 1024,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  bool metal_enabled() const {
#ifdef CC_HAVE_METAL
    return metal_ != nullptr;
#else
    return false;
#endif
  }

  bool cuda_enabled() const {
#ifdef CC_HAVE_CUDA
    return cuda_ != nullptr;
#else
    return false;
#endif
  }

  bool gpu_enabled() const { return metal_enabled() || cuda_enabled(); }

  const cc::ChesskersNet& net() const { return net_; }

  // One fused batched forward over the whole minibatch.
  EvalBatchResult EvalBatch(
      const std::vector<std::vector<float>>& positions,
      const std::vector<std::vector<std::vector<float>>>& moves_per) const {
#ifdef CC_HAVE_METAL
    if (metal_) {
      // The MPSGraph / Metal command queue is single-threaded: serialize GPU
      // submissions across search threads (the reference funnels all forwards
      // through one gatherer for the same reason). The minibatch is the
      // parallelism; the CPU value/gather heads inside eval_batch are per-board.
      std::lock_guard<std::mutex> lk(metal_mu_);
      return metal_->eval_batch(positions, moves_per);
    }
#endif
#ifdef CC_HAVE_CUDA
    if (cuda_) {
      // cuBLAS handle on the default stream: serialize GPU submissions across search
      // threads (the minibatch is the parallelism; the CPU value/gather heads inside
      // eval_batch are per-board, exactly as in the Metal path above).
      std::lock_guard<std::mutex> lk(cuda_mu_);
      return cuda_->eval_batch(positions, moves_per);
    }
#endif
    // CPU BLAS forward is const + re-entrant: safe to call from many threads.
    return net_.eval_batch(positions, moves_per);
  }

 private:
  cc::ChesskersNet net_;
#ifdef CC_HAVE_METAL
  std::unique_ptr<cc::MetalTrunkV2> metal_;
  mutable std::mutex metal_mu_;
#endif
#ifdef CC_HAVE_CUDA
  std::unique_ptr<cc::CudaTrunkV2> cuda_;
  mutable std::mutex cuda_mu_;
#endif
};

class ChesskersComputation : public BackendComputation {
 public:
  explicit ChesskersComputation(const ChesskersBackend& backend)
      : backend_(backend) {}

  size_t UsedBatchSize() const override { return items_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    Item item;
    // Chessckers encoding uses only the current board (no chess history planes).
    item.board = pos.pos.back().GetBoard();
    item.moves.assign(pos.legal_moves.begin(), pos.legal_moves.end());
    item.out = result;
    items_.push_back(std::move(item));
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    const size_t k = items_.size();
    if (k == 0) return;
    const cc::ChesskersNet& net = backend_.net();

    std::vector<std::vector<float>> positions;
    std::vector<std::vector<std::vector<float>>> moves_per;
    positions.reserve(k);
    moves_per.reserve(k);
    for (const auto& item : items_) {
      positions.push_back(net.is_v2 ? cc::encode_position_v2(item.board.cc())
                                    : cc::encode_position(item.board.cc()));
      std::vector<std::vector<float>> menc;
      menc.reserve(item.moves.size());
      for (const auto& m : item.moves) {
        menc.push_back(cc::encode_native_move(net, *m.native()));
      }
      moves_per.push_back(std::move(menc));
    }

    const EvalBatchResult results = backend_.EvalBatch(positions, moves_per);

    for (size_t i = 0; i < k; ++i) {
      const float value = results[i].first;
      const std::vector<float>& priors = results[i].second;
      EvalResultPtr& out = items_[i].out;
      if (out.q) *out.q = value;
      if (out.d) *out.d = 0.0f;  // has_wdl=false
      if (out.m) *out.m = 0.0f;  // has_mlh=false
      const size_t n = std::min(priors.size(), out.p.size());
      for (size_t j = 0; j < n; ++j) out.p[j] = priors[j];
    }
    items_.clear();
  }

 private:
  struct Item {
    ChessBoard board;
    MoveList moves;
    EvalResultPtr out;
  };
  const ChesskersBackend& backend_;
  std::vector<Item> items_;
};

std::unique_ptr<BackendComputation> ChesskersBackend::CreateComputation() {
  return std::make_unique<ChesskersComputation>(*this);
}

class ChesskersBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return 1000; }  // highest -> default pick
  std::string_view GetName() const override { return "chessckers"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<ChesskersBackend>(options);
  }
};

static BackendManager::Register reg_chessckers(
    std::make_unique<ChesskersBackendFactory>());

}  // namespace
}  // namespace lczero
