// Standalone parity test for the Chessckers ChessBoard/Move/Position adapter.
// Compiles ONLY the new chess core (types/board/position) + the vendored cc:: core
// (not the rest of lc0), and checks the adapter against direct cc:: calls — i.e.
// that the lc0-facing wrapper preserves the reference engine's behavior.
//
// Build (from repo root):
//   clang++ -std=c++20 -O2 -Isrc -Isrc/chessckers -DACCELERATE_NEW_LAPACK \
//     tests_chessckers/board_adapter_test.cc src/chess/board.cc src/chess/position.cc \
//     -framework Accelerate -o /tmp/cc_adapter_test && /tmp/cc_adapter_test

#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "chessckers/apply.hpp"
#include "chessckers/native_move.hpp"

using namespace lczero;

static int g_fail = 0;
#define CHECK(cond, msg)                                          \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("  FAIL: %s\n", msg);                           \
      ++g_fail;                                                   \
    }                                                             \
  } while (0)

// All UCIs the adapter generates for a FEN.
static std::set<std::string> AdapterMoveUcis(const ChessBoard& b) {
  std::set<std::string> s;
  for (const Move& m : b.GenerateLegalMoves()) s.insert(m.ToString());
  return s;
}
// All UCIs direct cc:: generates for the same FEN.
static std::set<std::string> CoreMoveUcis(const cc::Board& b) {
  std::set<std::string> s;
  for (const auto& nm : cc::gen_legal_native(b)) s.insert(nm.uci);
  return s;
}

static void TestFen(const std::string& fen) {
  std::printf("FEN: %s\n", fen.c_str());
  ChessBoard ab(fen);
  cc::Board cb = cc::parse_fen(fen);

  // 1. FEN round-trip: adapter serialize == cc serialize.
  CHECK(BoardToFen(ab) == cc::serialize_fen(cb), "FEN round-trip adapter==cc");

  // 2. Legal-move sets identical (adapter wraps cc, must match exactly).
  auto a = AdapterMoveUcis(ab);
  auto c = CoreMoveUcis(cb);
  CHECK(a == c, "legal-move UCI set adapter==cc");
  std::printf("  legal moves: %zu\n", a.size());

  // 3. Apply each adapter move; resulting FEN must match cc::apply_native.
  int checked = 0;
  for (const auto& nm : cc::gen_legal_native(cb)) {
    ChessBoard ab2 = ab;
    Move m = ab2.ParseMove(nm.uci);
    CHECK(!m.is_null(), "ParseMove resolves a generated uci");
    if (m.is_null()) continue;
    CHECK(m.ToString() == nm.uci, "Move::ToString round-trips uci");
    ab2.ApplyMove(m);
    cc::Board cb2 = cc::apply_native(cb, nm);
    CHECK(BoardToFen(ab2) == cc::serialize_fen(cb2), "applied FEN adapter==cc");
    if (++checked >= 12) break;  // sample to keep it fast
  }

  // 4. ComputeGameResult agrees with cc::detect_status mapping.
  PositionHistory hist;
  hist.Reset(Position::FromFen(fen));
  GameResult gr = hist.ComputeGameResult();
  cc::Status st = cc::detect_status(cb);
  GameResult expect = GameResult::UNDECIDED;
  if (st.winner == "white") expect = GameResult::WHITE_WON;
  else if (st.winner == "black") expect = GameResult::BLACK_WON;
  else if (st.status == "stalemate") expect = GameResult::DRAW;
  CHECK(gr == expect, "ComputeGameResult matches detect_status");
}

int main() {
  const char* kStart = ChessBoard::kStartposFen;

  // a) startpos
  TestFen(kStart);

  // b) startpos after White's first sub-move of the opening double-move
  {
    ChessBoard b(kStart);
    Move m = b.ParseMove("e2e4");
    if (!m.is_null()) {
      b.ApplyMove(m);
      TestFen(BoardToFen(b));
    } else {
      std::printf("NOTE: e2e4 not legal at startpos? (double-move ordering)\n");
    }
  }

  // c) a deeper position reached by playing a few plies through the adapter
  {
    ChessBoard b(kStart);
    PositionHistory hist;
    hist.Reset(Position::FromFen(kStart));
    int plies = 0;
    while (plies < 8) {
      auto moves = b.GenerateLegalMoves();
      if (moves.empty()) break;
      // Deterministic pick: lexicographically smallest uci.
      Move best = moves[0];
      for (auto& m : moves)
        if (m.ToString() < best.ToString()) best = m;
      b.ApplyMove(best);
      hist.Append(best);
      ++plies;
    }
    std::printf("played %d plies through adapter, result=%d\n", plies,
                static_cast<int>(hist.ComputeGameResult()));
    TestFen(BoardToFen(b));
  }

  // d) Position::FromFen counters sanity (startpos: White to move, ply 0).
  {
    Position p = Position::FromFen(kStart);
    CHECK(!p.IsBlackToMove(), "startpos is White to move");
    CHECK(p.GetGamePly() == 0, "startpos game ply == 0");
  }

  if (g_fail == 0)
    std::printf("\nALL ADAPTER PARITY CHECKS PASSED\n");
  else
    std::printf("\n%d CHECK(S) FAILED\n", g_fail);
  return g_fail ? 1 : 0;
}
