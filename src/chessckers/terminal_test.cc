// Regression test for Chessckers terminal scoring in the lc0 classic search.
//
// Guards cc::node_terminal() / cc::detect_status_known() (chessckers/apply.hpp),
// which search/classic/search.cc's ExtendNode() uses to score terminal leaves.
// The bug this guards: ExtendNode previously used the stock chess
// checkmate/stalemate test (legal_moves.empty() + IsUnderCheck()), which scored
// a Black self-stalemate as a DRAW instead of a loss, never recognized rank-8 /
// elimination wins, and could not tell a self-elimination from an opponent win.
//
// Standalone on purpose: the stock lc0 gtest suite (board_test.cc, ...) tests
// FIDE chess, which this fork does not play, so it is not built or reused here.

#include <cstdio>
#include <string>

#include "chessckers/apply.hpp"

using namespace cc;

static int g_failures = 0;

static size_t legal_count(const Board& b) {
    if (b.turn_white) {
        const WhiteBoard wb{b.occupied(),      b.occupied_white, b.pawns,  b.knights,
                            b.bishops,         b.rooks,          b.queens, b.kings,
                            b.castling_rights, (long)b.ep_square};
        return white_legal_moves(wb, b.stacks).size();
    }
    const uint64_t wk = b.kings & b.occupied_white;
    const int king_sq = wk ? __builtin_ctzll(wk) : -1;
    return all_black_legal_moves(b.occupied(), b.occupied_white, king_sq, b.stacks).size();
}

static const char* name(NodeTerminal t) {
    switch (t) {
        case NodeTerminal::kWin: return "kWin";
        case NodeTerminal::kLoss: return "kLoss";
        case NodeTerminal::kDraw: return "kDraw";
        case NodeTerminal::kNone: return "kNone";
    }
    return "?";
}

static void expect(const char* desc, const std::string& fen, NodeTerminal want) {
    const Board b = parse_fen(fen);
    const NodeTerminal got = node_terminal(b, legal_count(b) > 0);
    const bool ok = got == want;
    printf("  [%s] %-24s want=%-6s got=%-6s  %s\n", ok ? "PASS" : "FAIL", desc,
           name(want), name(got), fen.c_str());
    if (!ok) ++g_failures;
}

int main() {
    printf("Chessckers terminal scoring regression test\n");

    // THE bug: Black to move with no legal move is a LOSS (White wins), not a
    // stalemate draw. In node frame, White just moved and won -> kWin (wl_=+1),
    // which negamaxes to a loss at Black's decision node.
    expect("black self-stalemate", "8/8/8/8/7K/2P5/1P6/p7[a1:s] b - - 0 1",
           NodeTerminal::kWin);

    // White (FIDE) stalemate IS a draw — stalemate is asymmetric in Chessckers.
    expect("white stalemate->draw", "7K/5k2/8/8/8/8/8/8[f7:kkk] w - - 0 1",
           NodeTerminal::kDraw);

    // White mated -> the player who just moved (Black) won.
    expect("white mated", "7K/5kk1/8/8/8/8/8/8[f7:kk,g7:kk] w - - 0 1",
           NodeTerminal::kWin);

    // Tower elimination with Black to move (White just captured the last tower).
    expect("black eliminated", "8/8/8/8/7K/8/8/8 b - - 0 1", NodeTerminal::kWin);

    // Self-elimination: Black just emptied its own stacks. White wins the game,
    // but the player who just moved (Black) lost -> kLoss (wl_=-1).
    expect("black self-eliminated", "8/8/8/8/7K/8/8/8 w - - 0 1",
           NodeTerminal::kLoss);

    // White king captured -> the player who just moved (Black) won.
    expect("white king captured", "8/8/8/8/8/8/8/p7[a1:s] w - - 0 1",
           NodeTerminal::kWin);

    // Ongoing position must not be flagged terminal.
    expect("game continues", "8/8/8/8/7K/8/8/k7[a1:k] b - - 0 1",
           NodeTerminal::kNone);

    if (g_failures) {
        printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    printf("OK: all %s\n", "cases passed");
    return 0;
}
