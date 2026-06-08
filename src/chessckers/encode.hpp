// Chessckers C++ engine — Slice 6c: native NN encoders.
//
// Byte-for-byte port of the Rust encode_position_bb / encode_move (lib.rs), i.e.
// chessckers_engine.encoding.encode_position / encode_move. Position is built
// straight from the Board's bitboards + stacks (14*8*8); move from its fields
// (240-dim). The f64-divide-then-narrow-to-f32 is preserved exactly so the
// planes match PyTorch/Rust bit-for-bit.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "board.hpp"

namespace cc {

constexpr int ENC_POS_C = 15;
constexpr int ENC_MOVE_D = 240;

// channels 8-12 for one tower at (x,y) from its pieces (bottom-to-top {s,S,k}).
inline void apply_tower_channels(std::vector<float>& out, int x, int y, const std::string& pieces) {
    const int height = (int)pieces.size();
    if (height == 0) return;
    int kings = 0, stones = 0;
    for (char c : pieces) {
        if (c == 'k') ++kings;
        else if (c == 's' || c == 'S') ++stones;
    }
    const int base = y * 8 + x;
    out[8 * 64 + base] = static_cast<float>(static_cast<double>(height) / 24.0);
    out[9 * 64 + base] = static_cast<float>(static_cast<double>(stones) / 24.0);
    out[10 * 64 + base] = static_cast<float>(static_cast<double>(kings) / 24.0);
    if (pieces[height - 1] == 's') out[11 * 64 + base] = 1.0f;
    if (height >= 2 && pieces[height - 2] == 'k') out[12 * 64 + base] = 1.0f;
}

inline std::vector<float> encode_position(const Board& b) {
    std::vector<float> out(ENC_POS_C * 64, 0.0f);
    auto set_bits = [&](uint64_t bb, int ch) {
        while (bb) {
            const int sq = __builtin_ctzll(bb);  // sq = rank*8+file == y*8+x
            bb &= bb - 1;
            out[ch * 64 + sq] = 1.0f;
        }
    };
    const uint64_t ow = b.occupied_white, ob = b.occupied_black;
    set_bits(b.pawns & ow, 0);
    set_bits(b.knights & ow, 1);
    set_bits(b.bishops & ow, 2);
    set_bits(b.rooks & ow, 3);
    set_bits(b.queens & ow, 4);
    set_bits(b.kings & ow, 5);
    set_bits(b.pawns & ob, 6);   // Black Stone-top (pawn bb)
    set_bits(b.kings & ob, 7);   // Black King-top (king bb)
    for (const auto& [sq, pieces] : b.stacks) apply_tower_channels(out, sq & 7, sq >> 3, pieces);
    if (!b.turn_white)
        for (int i = 13 * 64; i < 14 * 64; ++i) out[i] = 1.0f;
    // Rank-8 win counter (#3): channel 14 = r8 / 3 (f64 divide -> narrow, matching
    // the Python reference), constant-filled.
    if (b.rank8_count != 0)
        for (int i = 14 * 64; i < 15 * 64; ++i)
            out[i] = static_cast<float>(static_cast<double>(b.rank8_count) / 3.0);
    return out;
}

// promo_index: matches the SINGLE-char codes the Rust uses; the dict's
// "promotion" is the full word ("queen"...), which never matches -> 0 (so every
// move sets channel 135). Preserved verbatim for exact parity.
inline int promo_index(const std::string& p) {
    if (p == "q") return 1;
    if (p == "r") return 2;
    if (p == "b") return 3;
    if (p == "n") return 4;
    return 0;
}

// 10x10 waypoint key -> (file10, rank10): rim files 'z'/'i' -> 0/9, 'a'..'h' ->
// 1..8; rank is the digit. None on malformed.
inline std::optional<std::pair<int, int>> file_rank10(const std::string& w) {
    if (w.size() != 2) return std::nullopt;
    int f;
    const char fc = w[0], rc = w[1];
    if (fc == 'z') f = 0;
    else if (fc >= 'a' && fc <= 'h') f = fc - 'a' + 1;
    else if (fc == 'i') f = 9;
    else return std::nullopt;
    if (rc < '0' || rc > '9') return std::nullopt;
    return std::make_pair(f, rc - '0');
}

inline std::vector<float> encode_move(int from_sq, int to_sq, bool has_capture,
                                      const std::vector<std::string>& waypoints, bool has_deploy,
                                      int deploy_count, bool has_demotions, int demotions_required,
                                      const std::string& promotion) {
    std::vector<float> out(ENC_MOVE_D, 0.0f);
    if (from_sq >= 0 && from_sq < 64) out[from_sq] = 1.0f;
    if (to_sq >= 0 && to_sq < 64) out[64 + to_sq] = 1.0f;
    if (has_capture) out[128] = 1.0f;
    if (!waypoints.empty()) out[129] = 1.0f;
    if (has_deploy) out[130] = 1.0f;
    if (has_demotions) out[131] = 1.0f;
    out[132] = static_cast<float>(static_cast<double>(waypoints.size()) / 8.0);
    out[133] = static_cast<float>(static_cast<double>(has_deploy ? deploy_count : 0) / 24.0);
    out[134] = static_cast<float>(static_cast<double>(has_demotions ? demotions_required : 0) / 8.0);
    out[135 + promo_index(promotion)] = 1.0f;
    for (const auto& w : waypoints) {
        const auto fr = file_rank10(w);
        if (fr) out[140 + fr->second * 10 + fr->first] = 1.0f;
    }
    return out;
}

// ===== V2 encoders (ChesskersScorerV2: 16ch 10x10 position + 114-d gather move) =====
// Byte-for-byte port of encoding.encode_position_v2 / encode_move_v2: the 15 V1
// channels written into the 8x8 interior of a 10x10 grid + an on-board mask plane
// (ch 15); moves carry 10x10 from/to indices (gathered against the spatial trunk),
// a rim-aware path mask (endpoints excluded), and 12 type scalars. f64-divide-then
// -narrow preserved so the planes match PyTorch bit-for-bit.

constexpr int ENC_POS_C_V2 = 16;
constexpr int ENC_MOVE_D_V2 = 114;

// 10x10 flat index (rank10*10 + file10) of an 8x8 board square — the interior cell,
// matching encoding._sq10 / _board_cell (the rim offsets the inner board by 1).
inline int sq10_of(int sq) { return ((sq >> 3) + 1) * 10 + ((sq & 7) + 1); }

inline std::vector<float> encode_position_v2(const Board& b) {
    std::vector<float> out(ENC_POS_C_V2 * 100, 0.0f);
    for (int r = 1; r <= 8; ++r)                                  // on-board mask (ch 15)
        for (int f = 1; f <= 8; ++f) out[15 * 100 + r * 10 + f] = 1.0f;
    auto set_bits = [&](uint64_t bb, int ch) {
        while (bb) {
            const int sq = __builtin_ctzll(bb);
            bb &= bb - 1;
            out[ch * 100 + sq10_of(sq)] = 1.0f;
        }
    };
    const uint64_t ow = b.occupied_white, ob = b.occupied_black;
    set_bits(b.pawns & ow, 0);
    set_bits(b.knights & ow, 1);
    set_bits(b.bishops & ow, 2);
    set_bits(b.rooks & ow, 3);
    set_bits(b.queens & ow, 4);
    set_bits(b.kings & ow, 5);
    set_bits(b.pawns & ob, 6);   // Black Stone-top
    set_bits(b.kings & ob, 7);   // Black King-top
    for (const auto& [sq, pieces] : b.stacks) {
        const int height = (int)pieces.size();
        if (height == 0) continue;
        int kings = 0, stones = 0;
        for (char c : pieces) { if (c == 'k') ++kings; else ++stones; }
        const int base = sq10_of(sq);
        out[8 * 100 + base] = static_cast<float>(static_cast<double>(height) / 24.0);
        out[9 * 100 + base] = static_cast<float>(static_cast<double>(stones) / 24.0);
        out[10 * 100 + base] = static_cast<float>(static_cast<double>(kings) / 24.0);
        if (pieces[height - 1] == 's') out[11 * 100 + base] = 1.0f;
        if (height >= 2 && pieces[height - 2] == 'k') out[12 * 100 + base] = 1.0f;
    }
    if (!b.turn_white)
        for (int r = 1; r <= 8; ++r)
            for (int f = 1; f <= 8; ++f) out[13 * 100 + r * 10 + f] = 1.0f;
    if (b.rank8_count != 0) {
        const float v = static_cast<float>(static_cast<double>(b.rank8_count) / 3.0);
        for (int r = 1; r <= 8; ++r)
            for (int f = 1; f <= 8; ++f) out[14 * 100 + r * 10 + f] = v;
    }
    return out;
}

inline int promo_index_v2(const std::string& p) {
    if (p == "q") return 1;
    if (p == "r") return 2;
    if (p == "b") return 3;
    if (p == "n") return 4;
    return 0;
}

inline std::vector<float> encode_move_v2(int from_sq, int to_sq,
                                         const std::vector<std::string>& waypoints, bool has_capture,
                                         bool has_deploy, int deploy_count, bool has_demotions,
                                         int demotions_required, const std::string& promotion) {
    std::vector<float> out(ENC_MOVE_D_V2, 0.0f);
    const bool from_ok = from_sq >= 0 && from_sq < 64, to_ok = to_sq >= 0 && to_sq < 64;
    const int fi = from_ok ? sq10_of(from_sq) : 11;
    const int ti = to_ok ? sq10_of(to_sq) : 11;
    out[0] = static_cast<float>(fi);
    out[1] = static_cast<float>(ti);
    for (const auto& w : waypoints) {
        const auto fr = file_rank10(w);
        if (fr) out[2 + fr->second * 10 + fr->first] = 1.0f;
    }
    if (from_ok) out[2 + fi] = 0.0f;   // endpoints gathered separately, never path cells
    if (to_ok) out[2 + ti] = 0.0f;
    const int s = 102;
    if (has_capture) out[s + 0] = 1.0f;
    if (!waypoints.empty()) out[s + 1] = 1.0f;
    if (has_deploy) out[s + 2] = 1.0f;
    if (has_demotions) out[s + 3] = 1.0f;
    out[s + 4] = static_cast<float>(static_cast<double>(waypoints.size()) / 8.0);
    out[s + 5] = static_cast<float>(static_cast<double>(has_deploy ? deploy_count : 0) / 24.0);
    out[s + 6] = static_cast<float>(static_cast<double>(has_demotions ? demotions_required : 0) / 8.0);
    out[s + 7 + promo_index_v2(promotion)] = 1.0f;
    return out;
}

}  // namespace cc
