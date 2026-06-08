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
    - run cc::ChesskersNet::eval -> (value, softmaxed priors),
    - write q=value (side-to-move WDL scalar, [-1,1]) and p[i]=priors[i].
  has_wdl/has_mlh are false for first parity (q only; no separate draw/MLH heads).
*/

#include <memory>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "chessckers/lc0_eval.hpp"
#include "chessckers/nn.hpp"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

class ChesskersBackend;

class ChesskersComputation : public BackendComputation {
 public:
  explicit ChesskersComputation(const cc::ChesskersNet& net) : net_(net) {}

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
    for (auto& item : items_) {
      auto [value, priors] = EvalChessckers(net_, item.board, item.moves);
      if (item.out.q) *item.out.q = value;
      if (item.out.d) *item.out.d = 0.0f;  // has_wdl=false: no separate draw head
      if (item.out.m) *item.out.m = 0.0f;  // has_mlh=false
      const size_t n = std::min(priors.size(), item.out.p.size());
      for (size_t i = 0; i < n; ++i) item.out.p[i] = priors[i];
    }
    items_.clear();
  }

 private:
  struct Item {
    ChessBoard board;
    MoveList moves;
    EvalResultPtr out;
  };
  const cc::ChesskersNet& net_;
  std::vector<Item> items_;
};

class ChesskersBackend : public Backend {
 public:
  explicit ChesskersBackend(const OptionsDict& options)
      : net_(options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    CERR << "Chessckers backend loaded net (is_v2=" << net_.is_v2 << ").";
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = false,
        .has_wdl = false,
        .runs_on_cpu = true,
        .suggested_num_search_threads = 2,
        .recommended_batch_size = 1,
        .maximum_batch_size = 1024,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<ChesskersComputation>(net_);
  }

 private:
  cc::ChesskersNet net_;
};

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
