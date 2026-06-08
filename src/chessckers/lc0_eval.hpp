// Chessckers port: the eval glue between lc0's game types and the cc:: net.
//
// Produces (value, per-move priors) for a position the way the reference engine's
// mcts_expand_pure does, but driven by lc0's ChessBoard + MoveList. The new lc0
// Backend (neural/backends/chessckers) wraps this to fill EvalResult{q, p[]}:
//   q = value (side-to-move WDL scalar W-L, in [-1,1]); p[i] = prior for legal[i].
// Priors are already softmaxed over the legal-move set (net.eval does the softmax),
// which matches what lc0's search stores as edge priors.
#pragma once

#include <utility>
#include <vector>

#include "chess/board.h"
#include "chessckers/encode.hpp"        // cc::encode_position[_v2]
#include "chessckers/native_move.hpp"   // cc::encode_native_move
#include "chessckers/nn.hpp"            // cc::ChesskersNet

namespace lczero {

inline std::pair<float, std::vector<float>> EvalChessckers(
    const cc::ChesskersNet& net, const ChessBoard& board,
    const MoveList& legal_moves) {
  const auto pos_enc = net.is_v2 ? cc::encode_position_v2(board.cc())
                                 : cc::encode_position(board.cc());
  std::vector<std::vector<float>> move_encs;
  move_encs.reserve(legal_moves.size());
  for (const auto& m : legal_moves) {
    move_encs.push_back(cc::encode_native_move(net, *m.native()));
  }
  return net.eval(pos_enc, move_encs);
}

}  // namespace lczero
