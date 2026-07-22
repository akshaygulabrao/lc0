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
  has_wdl is false (q only; no separate draw head). has_mlh is true when the net carries a
  moves-left head: its M (expected plies-to-end) feeds lc0's Q-gated moves-left search effect,
  the lc0-idiomatic "mate faster when already winning" signal (replaces the value discount).

  Batching is the lc0 GPU play: the search gathers a minibatch (recommended_batch_
  size below), AddInput is called per leaf, then ComputeBlocking() runs ONE batched
  forward. On Apple with CC_HAVE_METAL the trunk runs on the GPU (~10x/board).
*/

#include <condition_variable>
#include <deque>
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
#include "utils/atomic_vector.h"
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

// Upper bound on a single fused forward (also reported as maximum_batch_size).
// The cross-game coalescer never merges past this many positions into one batch.
constexpr size_t kCoalesceMaxBatch = 1024;

// One gathered eval input (see ChesskersComputation::AddInput). Board went POD
// in the 2026-07 stacks refactor, so an Item is ~1KB and a full
// kCoalesceMaxBatch AtomicVector is ~1MB — allocated per COMPUTATION, i.e.
// hundreds of times per second under GPU search. Those buffers are pooled on
// the backend (AcquireItems/ReleaseItems) instead of heap-cycled: glibc's
// dynamic mmap threshold otherwise adapts past 1MB and the freed buffers are
// retained in the arenas (~165MB/s RSS growth), while pinning the threshold
// (see main.cc) makes each allocation a fresh zero-filled mmap — a measurable
// page-fault tax. Reuse avoids both.
struct Item {
    ChessBoard board;
    MoveList moves;
    EvalResultPtr out;
};
using ItemVec = AtomicVector<Item>;

class ChesskersBackend : public Backend {
 public:
  explicit ChesskersBackend(const OptionsDict& options)
      : net_(options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    // CC_FORCE_CPU=1 skips the GPU trunks and runs the CPU BLAS forward — used to
    // parity-check the GPU backends against the CPU oracle (e.g. for the V4 SE port).
    const bool force_cpu = std::getenv("CC_FORCE_CPU") != nullptr;
#ifdef CC_HAVE_METAL
    if (!force_cpu && net_.is_v2) {
      auto metal = std::make_unique<cc::MetalTrunkV2>(net_);
      if (metal->ok()) {
        metal_ = std::move(metal);
        CERR << "Chessckers backend: Metal GPU trunk enabled (is_v2="
             << net_.is_v2 << ").";
      }
    }
#endif
#ifdef CC_HAVE_CUDA
    if (!force_cpu && !metal_enabled() && net_.is_v2) {
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
        // Moves-left effect on iff the loaded net actually has the head (older nets predate it).
        .has_mlh = net_.is_v2 && net_.has_moves_left,
        .has_wdl = false,
        .runs_on_cpu = !gpu_enabled(),
        .suggested_num_search_threads = 2,
        // Encourage the search to gather a real minibatch so the batched/GPU
        // trunk pays off (the conv GEMMs fuse across the batch).
        .recommended_batch_size = 256,
        .maximum_batch_size = static_cast<int>(kCoalesceMaxBatch),
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

  // One fused batched forward over the whole minibatch. m_out (optional): per-board moves-left
  // (expected plies-to-end) when the net has the head; forwarded to the active backend.
  EvalBatchResult EvalBatch(
      const std::vector<std::vector<float>>& positions,
      const std::vector<std::vector<std::vector<float>>>& moves_per,
      std::vector<float>* m_out = nullptr) const {
#ifdef CC_HAVE_METAL
    if (metal_) {
      // The MPSGraph / Metal command queue is single-threaded: serialize GPU
      // submissions across search threads (the reference funnels all forwards
      // through one gatherer for the same reason). The minibatch is the
      // parallelism; the CPU value/gather heads inside eval_batch are per-board.
      std::lock_guard<std::mutex> lk(metal_mu_);
      return metal_->eval_batch(positions, moves_per, m_out);
    }
#endif
#ifdef CC_HAVE_CUDA
    if (cuda_) {
      // ONE GPU, legacy default stream: serialize CUDA eval across ALL backend
      // instances (lc0 builds several), not just this one. A per-instance mutex
      // left concurrent host-side CUDA work (cudaMalloc in DBuf::ensure, cuBLAS
      // workspace) from sibling instances racing on the shared default stream —
      // an illegal-memory-access under load. The device work is already
      // serialized on stream 0, so this process-wide lock costs ~nothing.
      static std::mutex cuda_global_mu;
      std::lock_guard<std::mutex> lk(cuda_global_mu);
      return cuda_->eval_batch(positions, moves_per, m_out);
    }
#endif
    // CPU BLAS forward is const + re-entrant: safe to call from many threads.
    return net_.eval_batch(positions, moves_per, m_out);
  }

  // Cross-game batch coalescing — the new-Backend-API analog of lc0's
  // "multiplexing" network (backends/network_mux.cc). Under --parallelism>1 each
  // self-play game's search thread arrives here with its own small minibatch;
  // uncoalesced they serialize as N tiny forwards on the GPU lock. Here one
  // "leader" thread drains ALL currently-queued submissions into a single fused
  // EvalBatch, so concurrent games share GPU batches. Per-backend-instance, hence
  // per-net: an arena's two nets use two backends -> two coalescers -> never mix
  // nets in one batch. With --parallelism=1 the queue is size 1: a passthrough.
  EvalBatchResult EvalCoalesced(
      const std::vector<std::vector<float>>& positions,
      const std::vector<std::vector<std::vector<float>>>& moves_per,
      std::vector<float>* m_out) const {
    // CPU BLAS eval is re-entrant and benefits from running concurrently across
    // search threads (each call can use multi-threaded BLAS); coalescing would
    // serialize that. Only coalesce on GPU, where the device lock already
    // serializes forwards so merging into one big batch is a pure win.
    if (!gpu_enabled()) return EvalBatch(positions, moves_per, m_out);

    CoalesceSub sub{&positions, &moves_per, m_out != nullptr, {}, {}, false};

    std::unique_lock<std::mutex> lk(coalesce_mu_);
    coalesce_queue_.push_back(&sub);
    while (!sub.done) {
      if (coalesce_processing_) {
        coalesce_cv_.wait(lk);
        continue;
      }
      // Become the leader: drain the queue (capped at the max batch) and run it.
      coalesce_processing_ = true;
      std::vector<CoalesceSub*> batch;
      size_t total = 0;
      while (!coalesce_queue_.empty()) {
        CoalesceSub* s = coalesce_queue_.front();
        const size_t n = s->positions->size();
        if (!batch.empty() && total + n > kCoalesceMaxBatch) break;
        batch.push_back(s);
        total += n;
        coalesce_queue_.pop_front();
      }
      lk.unlock();

      // Merge -> ONE forward -> scatter, OUTSIDE the queue lock so sibling games
      // keep enqueueing while this batch computes. EvalBatch holds the GPU lock.
      try {
        bool want_mlh = false;
        std::vector<std::vector<float>> merged_pos;
        std::vector<std::vector<std::vector<float>>> merged_moves;
        merged_pos.reserve(total);
        merged_moves.reserve(total);
        for (CoalesceSub* s : batch) {
          want_mlh = want_mlh || s->want_mlh;
          merged_pos.insert(merged_pos.end(), s->positions->begin(),
                            s->positions->end());
          merged_moves.insert(merged_moves.end(), s->moves_per->begin(),
                              s->moves_per->end());
        }
        std::vector<float> merged_mls;
        const EvalBatchResult merged = EvalBatch(
            merged_pos, merged_moves, want_mlh ? &merged_mls : nullptr);
        size_t off = 0;
        for (CoalesceSub* s : batch) {
          const size_t n = s->positions->size();
          s->results.assign(merged.begin() + off, merged.begin() + off + n);
          if (s->want_mlh && merged_mls.size() >= off + n) {
            s->mls.assign(merged_mls.begin() + off, merged_mls.begin() + off + n);
          }
          off += n;
        }
      } catch (...) {
        // Never leave siblings deadlocked on a failed forward: wake them (with
        // empty results) and re-throw on this thread. A GPU fault here is fatal
        // anyway; this surfaces it instead of hanging the other games.
        lk.lock();
        for (CoalesceSub* s : batch) s->done = true;
        coalesce_processing_ = false;
        coalesce_cv_.notify_all();
        throw;
      }

      lk.lock();
      for (CoalesceSub* s : batch) s->done = true;
      coalesce_processing_ = false;
      coalesce_cv_.notify_all();
    }

    if (m_out) *m_out = std::move(sub.mls);
    return std::move(sub.results);
  }

 private:
  // One queued evaluation request awaiting coalescing (see EvalCoalesced). The
  // input pointers alias the caller's stack, valid until done (the caller blocks).
  struct CoalesceSub {
    const std::vector<std::vector<float>>* positions;
    const std::vector<std::vector<std::vector<float>>>* moves_per;
    bool want_mlh;
    EvalBatchResult results;
    std::vector<float> mls;
    bool done;
  };
  // mutable: EvalCoalesced is const (invoked via a const backend ref), like the
  // GPU mutexes below.
  mutable std::mutex coalesce_mu_;
  mutable std::condition_variable coalesce_cv_;
  mutable std::deque<CoalesceSub*> coalesce_queue_;
  mutable bool coalesce_processing_ = false;

 public:
  // Item-buffer pool (see the Item comment above). const because computations
  // hold a const backend ref; the pool is mutable like the GPU mutexes.
  std::unique_ptr<ItemVec> AcquireItems() const {
    {
      std::lock_guard<std::mutex> lk(pool_mu_);
      if (!item_pool_.empty()) {
        auto v = std::move(item_pool_.back());
        item_pool_.pop_back();
        return v;
      }
    }
    return std::make_unique<ItemVec>(kCoalesceMaxBatch);
  }
  void ReleaseItems(std::unique_ptr<ItemVec> v) const {
    v->clear();
    std::lock_guard<std::mutex> lk(pool_mu_);
    item_pool_.push_back(std::move(v));
  }

 private:
  mutable std::mutex pool_mu_;
  mutable std::vector<std::unique_ptr<ItemVec>> item_pool_;

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
      : backend_(backend), items_(backend.AcquireItems()) {}

  ~ChesskersComputation() override { backend_.ReleaseItems(std::move(items_)); }

  size_t UsedBatchSize() const override { return items_->size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    // Called CONCURRENTLY by the classic search's task workers (ProcessPickedTask
    // fans leaf-picking across threads on GPU backends), so the container must be
    // an AtomicVector like upstream's NetworkAsBackendComputation — a plain
    // vector push_back here segfaults under multi-threaded gathering.
    Item item;
    // Chessckers encoding uses only the current board (no chess history planes).
    item.board = pos.pos.back().GetBoard();
    item.moves.assign(pos.legal_moves.begin(), pos.legal_moves.end());
    item.out = result;
    items_->emplace_back(std::move(item));
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    const size_t k = items_->size();
    if (k == 0) return;
    const cc::ChesskersNet& net = backend_.net();

    std::vector<std::vector<float>> positions;
    std::vector<std::vector<std::vector<float>>> moves_per;
    positions.reserve(k);
    moves_per.reserve(k);
    for (const auto& item : *items_) {
      positions.push_back(net.is_v2 ? cc::encode_position_v2(item.board.cc())
                                    : cc::encode_position(item.board.cc()));
      std::vector<std::vector<float>> menc;
      menc.reserve(item.moves.size());
      for (const auto& m : item.moves) {
        menc.push_back(cc::encode_native_move(net, *m.native()));
      }
      moves_per.push_back(std::move(menc));
    }

    // Request moves-left only when the net has the head; else leave M at 0 (lc0 won't use it).
    std::vector<float> mls;
    const bool want_mlh = net.is_v2 && net.has_moves_left;
    const EvalBatchResult results =
        backend_.EvalCoalesced(positions, moves_per, want_mlh ? &mls : nullptr);

    for (size_t i = 0; i < k; ++i) {
      const float value = results[i].first;
      const std::vector<float>& priors = results[i].second;
      EvalResultPtr& out = (*items_)[i].out;
      if (out.q) *out.q = value;
      if (out.d) *out.d = 0.0f;  // has_wdl=false
      if (out.m) *out.m = (want_mlh && i < mls.size()) ? mls[i] : 0.0f;
      const size_t n = std::min(priors.size(), out.p.size());
      for (size_t j = 0; j < n; ++j) out.p[j] = priors[j];
    }
    items_->clear();
  }

 private:
  const ChesskersBackend& backend_;
  // Pooled; capacity = kCoalesceMaxBatch = the advertised maximum_batch_size:
  // the search never gathers more inputs than that into one computation.
  std::unique_ptr<ItemVec> items_;
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
