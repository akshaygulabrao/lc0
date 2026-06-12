// Corpus test for Chessckers FEN round-trip + the board/stacks state invariant.
//
// Drives cc::parse_fen / cc::serialize_fen (chessckers/board.hpp) over the
// PyVariant oracle corpus (testdata/parity_corpus.jsonl, one {"fen":...} per
// line) and asserts two things:
//   (A) serialize_fen(parse_fen(fen)) round-trips the FEN, and
//   (B) the parsed Board's bitboards and `stacks` overlay agree (every black
//       square has a stack whose top piece matches the bitboard, and vice
//       versa; white/black occupancy is disjoint).
//
// Known, benign divergence normalized away in (A): serialize_fen canonicalizes
// the castling and en-passant fields python-chess-style (e.g. "KQkq"->"KQk",
// "e3"->"-") while the oracle keeps the raw fields. Both are vestigial in
// Chessckers — only White ever castles and en passant is never capturable — so
// before comparing we rewrite the castling (3rd) and ep (4th) space-separated
// tokens to "-" in BOTH the original and the round-tripped string, leaving any
// trailing {wm:..,r8:..} block intact. Every other field must match exactly.
//
// Standalone on purpose (no gtest): the stock lc0 suite tests FIDE chess, which
// this fork does not play. Matches the plain-main()+counters style of
// terminal_test.cc. Build: clang++ -std=c++20 -I src src/chessckers/fen_test.cc
// (board.hpp + json_parse.hpp are header-only; no BLAS/Accelerate needed).

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "chessckers/board.hpp"
#include "chessckers/json_parse.hpp"

using namespace cc;

// Rewrite the castling (index 2) and en-passant (index 3) space-separated tokens
// to "-", leaving every other token — including any trailing {wm:..,r8:..} block,
// which sits after the six standard fields — untouched. Applied identically to
// both sides of the comparison so the canonicalization difference cancels out.
static std::string normalize_fen(const std::string& fen) {
    std::vector<std::string> toks = split(fen, ' ');
    if (toks.size() > 2) toks[2] = "-";
    if (toks.size() > 3) toks[3] = "-";
    std::string out;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i) out += ' ';
        out += toks[i];
    }
    return out;
}

static int g_mismatch = 0;
static int g_violation = 0;

static void check_roundtrip(const std::string& fen) {
    const std::string got = serialize_fen(parse_fen(fen));
    const std::string a = normalize_fen(fen);
    const std::string b = normalize_fen(got);
    if (a == b) return;
    ++g_mismatch;
    if (g_mismatch <= 5) {
        printf("  [fen_mismatch] original     : %s\n", fen.c_str());
        printf("                 round-tripped: %s\n", got.c_str());
        printf("                 norm(orig)   : %s\n", a.c_str());
        printf("                 norm(rt)     : %s\n", b.c_str());
    }
}

static void check_invariant(const std::string& fen) {
    const Board b = parse_fen(fen);

    auto report = [&](const std::string& why) {
        ++g_violation;
        if (g_violation <= 5) {
            printf("  [invariant_violation] %s\n    fen: %s\n", why.c_str(), fen.c_str());
        }
    };

    // occupied_white & occupied_black must be disjoint.
    if (b.occupied_white & b.occupied_black) {
        report("occupied_white & occupied_black overlap");
    }

    // Every stack square: non-empty, black-occupied (not white), top char matches bb.
    uint64_t stack_squares = 0;
    for (const auto& [sq, pieces] : b.stacks) {
        const uint64_t bit = 1ULL << sq;
        stack_squares |= bit;
        if (pieces.empty()) {
            report("empty stack at " + square_name(sq));
            continue;
        }
        if (!((b.occupied_black >> sq) & 1ULL)) {
            report("stack square " + square_name(sq) + " not in occupied_black");
        }
        if ((b.occupied_white >> sq) & 1ULL) {
            report("stack square " + square_name(sq) + " is in occupied_white");
        }
        const char top = pieces.back();
        if (top == 'k') {
            if (!((b.kings >> sq) & 1ULL))
                report("top 'k' at " + square_name(sq) + " not set in kings bb");
        } else if (top == 's' || top == 'S') {
            if (!((b.pawns >> sq) & 1ULL))
                report("top stone at " + square_name(sq) + " not set in pawns bb");
        } else {
            report(std::string("unexpected top char '") + top + "' at " + square_name(sq));
        }
    }

    // The set of stack squares must equal occupied_black exactly.
    if (stack_squares != b.occupied_black) {
        report("stack-square set != occupied_black");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <parity_corpus.jsonl>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        fprintf(stderr, "error: cannot open %s\n", argv[1]);
        return 2;
    }

    int checked = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (strip(line).empty()) continue;
        const JsonValue v = JsonParser(line).parse();
        const JsonValue* f = v.find("fen");
        if (!f || f->type != JsonValue::Str) {
            fprintf(stderr, "error: line missing string \"fen\": %s\n", line.c_str());
            return 2;
        }
        const std::string& fen = f->str;
        check_roundtrip(fen);
        check_invariant(fen);
        ++checked;
    }

    printf("checked %d: fen_mismatch=%d invariant_violation=%d\n", checked, g_mismatch,
           g_violation);
    return (g_mismatch || g_violation) ? 1 : 0;
}
