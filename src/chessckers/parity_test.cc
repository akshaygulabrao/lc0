// Parity test: C++ rules port vs the PyVariant oracle corpus.
//
// PyVariant (the separate Python rules repo) is the Chessckers rules authority;
// this C++ fork must match it exactly. This test replays an oracle corpus
// (JSONL, one position per line) and asserts the C++ port agrees on both
//   (a) the legal move set (cc::gen_legal_native, chessckers/native_move.hpp), and
//   (b) the game status / winner (cc::detect_status, chessckers/apply.hpp).
// Each corpus line is:
//   {"fen":"<chessckers FEN>","legal":[<uci>,...sorted],
//    "status":"<''|variantEnd|mate|stalemate>","winner":"<''|white|black>"}
//
// Standalone on purpose (plain main() + counters, no gtest): the stock lc0 gtest
// suite (board_test.cc, ...) tests FIDE chess, which this fork does not play, so
// it is not built or reused here. Run as:
//   parity_test <path-to-parity_corpus.jsonl> [bench_loops]
// With bench_loops > 0, after the parity pass it re-runs gen_legal_native over
// the whole (pre-parsed) corpus that many times and prints positions/sec —
// a CPU-bound movegen benchmark with parse/IO excluded from the timed region.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "chessckers/apply.hpp"
#include "chessckers/json_parse.hpp"
#include "chessckers/native_move.hpp"

using namespace cc;

// Print a capped sample of strings from a vector (keeps mismatch diffs readable).
static void print_capped(const char* label, const std::vector<std::string>& v, size_t cap) {
    printf("      %s (%zu):", label, v.size());
    for (size_t i = 0; i < v.size() && i < cap; ++i) printf(" %s", v[i].c_str());
    if (v.size() > cap) printf(" ...");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <parity_corpus.jsonl>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        printf("error: cannot open corpus %s\n", argv[1]);
        return 2;
    }

    const int bench_loops = (argc >= 3) ? std::atoi(argv[2]) : 0;
    std::vector<Board> boards;

    int checked = 0;
    int gen_mismatch = 0;
    int status_mismatch = 0;
    const int kReportCap = 5;  // cap detailed mismatch dumps at the first ~5 of each kind

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const JsonValue v = JsonParser(line).parse();

        const JsonValue* fen_v = v.find("fen");
        const JsonValue* legal_v = v.find("legal");
        const JsonValue* status_v = v.find("status");
        const JsonValue* winner_v = v.find("winner");
        const std::string fen = fen_v ? fen_v->str : "";
        const std::string exp_status = status_v ? status_v->str : "";
        const std::string exp_winner = winner_v ? winner_v->str : "";

        std::vector<std::string> expected;  // already sorted in the corpus
        if (legal_v) {
            for (const JsonValue& e : legal_v->arr) expected.push_back(e.str);
        }

        const Board b = parse_fen(fen);
        if (bench_loops > 0) boards.push_back(b);

        // (a) legal move set: collect every generated uci, sort, compare for exact equality.
        std::vector<std::string> got;
        for (const NativeMove& m : gen_legal_native(b)) got.push_back(m.uci);
        std::sort(got.begin(), got.end());

        if (got != expected) {
            ++gen_mismatch;
            if (gen_mismatch <= kReportCap) {
                printf("GEN MISMATCH: %s\n", fen.c_str());
                // Diff (both inputs sorted): missing = in expected not got; extra = in got not expected.
                std::vector<std::string> missing, extra;
                std::set_difference(expected.begin(), expected.end(), got.begin(), got.end(),
                                    std::back_inserter(missing));
                std::set_difference(got.begin(), got.end(), expected.begin(), expected.end(),
                                    std::back_inserter(extra));
                print_capped("missing", missing, kReportCap);
                print_capped("extra", extra, kReportCap);
            }
        }

        // (b) status + winner.
        const Status st = detect_status(b);
        if (st.status != exp_status || st.winner != exp_winner) {
            ++status_mismatch;
            if (status_mismatch <= kReportCap) {
                printf("STATUS MISMATCH: %s\n", fen.c_str());
                printf("      want status='%s' winner='%s'  got status='%s' winner='%s'\n",
                       exp_status.c_str(), exp_winner.c_str(), st.status.c_str(),
                       st.winner.c_str());
            }
        }

        ++checked;
    }

    printf("checked %d: gen_mismatch=%d status_mismatch=%d\n", checked, gen_mismatch,
           status_mismatch);

    if (bench_loops > 0) {
        size_t total_moves = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int l = 0; l < bench_loops; ++l)
            for (const Board& b : boards) total_moves += gen_legal_native(b).size();
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double pos = (double)bench_loops * boards.size();
        printf("bench: %d loops x %zu positions = %.0f pos/s (%.2fs, %zu moves)\n",
               bench_loops, boards.size(), pos / secs, secs, total_moves);
    }
    return (gen_mismatch || status_mismatch) ? 1 : 0;
}
