// Standalone eval-path test: loads a real cc:: net and verifies the lc0 adapter
// eval glue (ChessBoard + MoveList -> value + priors) is byte-identical to the
// reference engine's direct path (gen_legal_native + encode_native_move + eval).
//
// Build (from repo root), passing a net .bin:
//   clang++ -std=c++20 -O2 -Isrc -Isrc/chessckers -DACCELERATE_NEW_LAPACK \
//     tests_chessckers/eval_test.cc src/chess/board.cc src/chess/position.cc \
//     -framework Accelerate -o /tmp/cc_eval_test && \
//   /tmp/cc_eval_test ../chessckers/engine/net-<hash>.bin

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chessckers/lc0_eval.hpp"
#include "chessckers/native_move.hpp"

using namespace lczero;

static int g_fail = 0;
#define CHECK(cond, msg)                            \
  do {                                              \
    if (!(cond)) {                                  \
      std::printf("  FAIL: %s\n", msg);             \
      ++g_fail;                                     \
    }                                               \
  } while (0)

static void TestFen(const cc::ChesskersNet& net, const std::string& fen) {
  std::printf("FEN: %s\n", fen.c_str());
  ChessBoard b(fen);
  MoveList moves = b.GenerateLegalMoves();

  // lc0 adapter eval path.
  auto [value, priors] = EvalChessckers(net, b, moves);

  // Direct reference path (same net, same FEN).
  cc::Board cb = cc::parse_fen(fen);
  auto legal = cc::gen_legal_native(cb);
  const auto pos_enc =
      net.is_v2 ? cc::encode_position_v2(cb) : cc::encode_position(cb);
  std::vector<std::vector<float>> menc;
  for (const auto& m : legal) menc.push_back(cc::encode_native_move(net, m));
  auto [rvalue, rpriors] = net.eval(pos_enc, menc);

  // 1. shapes
  CHECK(priors.size() == moves.size(), "priors aligned to legal moves");
  CHECK(priors.size() == rpriors.size(), "adapter/ref same #priors");

  // 2. byte-identical value + priors (glue must reproduce the reference exactly)
  CHECK(value == rvalue, "adapter value == reference value");
  bool priors_eq = priors.size() == rpriors.size();
  for (size_t i = 0; i < priors.size() && priors_eq; ++i)
    if (priors[i] != rpriors[i]) priors_eq = false;
  CHECK(priors_eq, "adapter priors == reference priors (byte-identical)");

  // 3. sanity: value in range, priors a valid distribution
  CHECK(value >= -1.0001f && value <= 1.0001f, "value in [-1,1]");
  double sum = 0;
  bool all_ok = true;
  for (float p : priors) {
    if (!std::isfinite(p) || p < -1e-6f) all_ok = false;
    sum += p;
  }
  CHECK(all_ok, "priors finite and non-negative");
  CHECK(std::abs(sum - 1.0) < 1e-3, "priors sum to 1");

  // report top move
  size_t best = 0;
  for (size_t i = 1; i < priors.size(); ++i)
    if (priors[i] > priors[best]) best = i;
  std::printf("  value=%.4f  moves=%zu  top=%s (p=%.4f)  is_v2=%d\n", value,
              moves.size(), moves[best].ToString().c_str(), priors[best],
              net.is_v2);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <net.bin>\n", argv[0]);
    return 2;
  }
  cc::ChesskersNet net(argv[1]);
  std::printf("loaded net: %s (is_v2=%d)\n\n", argv[1], net.is_v2);

  TestFen(net, ChessBoard::kStartposFen);

  // a mid-game-ish position reached by a few adapter plies
  ChessBoard b(ChessBoard::kStartposFen);
  for (int i = 0; i < 6; ++i) {
    auto mv = b.GenerateLegalMoves();
    if (mv.empty()) break;
    Move pick = mv[0];
    for (auto& m : mv)
      if (m.ToString() < pick.ToString()) pick = m;
    b.ApplyMove(pick);
  }
  TestFen(net, BoardToFen(b));

  if (g_fail == 0)
    std::printf("\nALL EVAL PARITY CHECKS PASSED\n");
  else
    std::printf("\n%d CHECK(S) FAILED\n", g_fail);
  return g_fail ? 1 : 0;
}
