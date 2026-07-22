// cc_chunk_dump — deterministic ccz1 golden generator for refactor byte-parity.
//
// Reads a parity-corpus jsonl (only the "fen" field is used), builds a synthetic
// but fully deterministic PureGame over the first N positions (visits/policy
// derived from index arithmetic, no RNG), routes half the records through the
// canonical-sort + legal-drop + hash path that game.cc uses in production, and
// prints the encoded chunk JSON to stdout. Diffing this output across a refactor
// proves the emission layer (uci strings, waypoint keys, chain fields, JSON
// shape) is byte-identical. Usage: cc_chunk_dump <corpus.jsonl> [N=200]
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "chessckers/apply.hpp"
#include "chessckers/board.hpp"
#include "chessckers/chunk.hpp"
#include "chessckers/native_move.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: cc_chunk_dump <corpus.jsonl> [N]\n");
        return 2;
    }
    const int want = argc > 2 ? std::atoi(argv[2]) : 200;

    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    cc::PureGame game;
    std::string line;
    int i = 0;
    while (i < want && std::getline(in, line)) {
        // minimal extraction of the "fen" field (first string value in the line)
        const size_t k = line.find("\"fen\":\"");
        if (k == std::string::npos) continue;
        const size_t s = k + 7;
        const size_t e = line.find('"', s);
        const std::string fen = line.substr(s, e - s);

        cc::Board b;
        try {
            b = cc::parse_fen(fen);
        } catch (...) {
            continue;
        }
        std::vector<cc::NativeMove> legal = cc::gen_legal_native(b);
        if (legal.empty()) continue;

        cc::PureRecord rec;
        rec.fen = fen;
        rec.side_white = b.turn_white;
        rec.ply = i;
        rec.root_q = (float)((i % 21) - 10) / 10.0f;
        rec.root_d = (float)(i % 11) / 10.0f;
        rec.visits.resize(legal.size());
        rec.improved_policy.resize(legal.size());
        double zsum = 0.0;
        for (size_t j = 0; j < legal.size(); ++j) {
            rec.visits[j] = (int)((j + i) % 7);
            rec.improved_policy[j] = (float)(1 + (j + 2 * i) % 5);
            zsum += rec.improved_policy[j];
        }
        for (auto& p : rec.improved_policy) p = (float)(p / zsum);
        rec.legal = std::move(legal);

        if (i % 2 == 0) {
            // Production path: canonical uci-sort, stamp hash, drop the move
            // list (encode_chunk regenerates + verifies). Mirrors game.cc.
            std::vector<size_t> perm(rec.legal.size());
            std::iota(perm.begin(), perm.end(), 0);
            std::sort(perm.begin(), perm.end(), [&](size_t a, size_t c) {
                return rec.legal[a].uci < rec.legal[c].uci;
            });
            std::vector<cc::NativeMove> legal_s(perm.size());
            std::vector<int> visits_s(perm.size());
            std::vector<float> policy_s(perm.size());
            for (size_t j = 0; j < perm.size(); ++j) {
                legal_s[j] = std::move(rec.legal[perm[j]]);
                visits_s[j] = rec.visits[perm[j]];
                policy_s[j] = rec.improved_policy[perm[j]];
            }
            rec.visits = std::move(visits_s);
            rec.improved_policy = std::move(policy_s);
            rec.legal_hash = cc::legal_ucis_hash(legal_s);
            rec.legal.clear();
            rec.legal.shrink_to_fit();
        }
        game.records.push_back(std::move(rec));
        ++i;
    }
    game.outcome = "black";
    game.total_plies = i;
    // encode_chunk returns gzipped binary — fwrite, not fputs (NULs inside).
    const std::string gz = cc::encode_chunk(game);
    std::fprintf(stderr, "records=%d bytes=%zu\n", (int)game.records.size(), gz.size());
    std::fwrite(gz.data(), 1, gz.size(), stdout);
    return 0;
}
