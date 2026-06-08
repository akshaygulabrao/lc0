// Chessckers C++ engine — Slice 3b: White move generation.
//
// 1:1 port of the Rust white_* functions. White plays standard FIDE chess, but:
//  - Black squares (encoded Black-Pawn / Black-King) are blockers/captures, and
//  - legality is filtered by the CHESSCKERS check predicate (movegen.hpp), not
//    FIDE check — python-chess's is_check mis-models the Black-King encoding.
// Castling additionally uses a FIDE attack model (black_attacks_square_fide) to
// replicate python-chess's castling pseudo-legal path conditions, then rejects
// crossed squares attacked under the Chessckers model.
//
// Held byte-equivalent to the Rust via tests/test_cpp_white_moves.py
// (exact-ordered vs the Rust extension over self-play rollouts + edge FENs).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "movegen.hpp"  // geometry, owner, square_name, the Chessckers check predicate

namespace cc {

constexpr std::pair<int, int> KNIGHT_DELTAS[8] = {{1, 2},   {2, 1},   {2, -1}, {1, -2},
                                                  {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
constexpr std::pair<int, int> KING_DELTAS[8] = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
constexpr std::pair<int, int> BISHOP_DIRS[4] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
constexpr std::pair<int, int> ROOK_DIRS[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

constexpr uint64_t WB_RANK_1 = 0x00000000000000FFULL;
constexpr uint64_t WB_RANK_4 = 0x00000000FF000000ULL;

enum class WPiece { Pawn, Knight, Bishop, Rook, Queen, King };

inline const char* wpiece_name(WPiece p) {
    switch (p) {
        case WPiece::Pawn: return "pawn";
        case WPiece::Knight: return "knight";
        case WPiece::Bishop: return "bishop";
        case WPiece::Rook: return "rook";
        case WPiece::Queen: return "queen";
        case WPiece::King: return "king";
    }
    return "?";
}

// Inverse of wpiece_name: piece word -> WPiece (default King, matching the
// bindings' historical fallback). Used by both the dict parse and the pure
// WCandidate->WhiteMove converter.
inline WPiece wpiece_from_name(const std::string& n) {
    if (n == "pawn") return WPiece::Pawn;
    if (n == "knight") return WPiece::Knight;
    if (n == "bishop") return WPiece::Bishop;
    if (n == "rook") return WPiece::Rook;
    if (n == "queen") return WPiece::Queen;
    return WPiece::King;
}

struct WCandidate {
    int from_sq, to_sq;
    WPiece piece;
    std::optional<std::string> promotion;  // "queen"/"rook"/"bishop"/"knight"
    int capture_sq;                        // -1 == none
    bool is_en_passant;
    bool is_castling;
    bool castling_kingside;  // valid iff is_castling
    int castling_rook_sq;    // valid iff is_castling
};

struct WhiteBoard {
    uint64_t occupied, occupied_white, pawns, knights, bishops, rooks, queens, kings, castling_rights;
    long ep_square;  // -1 == none
    uint64_t occupied_black() const { return occupied & ~occupied_white; }
    int white_king_sq() const {
        const uint64_t wk = kings & occupied_white;
        return wk ? __builtin_ctzll(wk) : -1;
    }
};

// FIDE attacker test (python-chess attackers_mask semantics): does a Black piece
// attack `target` given a hypothetical `occupied`? Used only for castling.
inline bool black_attacks_square_fide(uint64_t occupied, uint64_t pawns, uint64_t knights,
                                      uint64_t bishops, uint64_t rooks, uint64_t queens,
                                      uint64_t kings, int target, uint64_t occ_black) {
    const int tf = target & 7, tr = target >> 3;
    for (auto [df, dr] : KNIGHT_DELTAS) {
        const int nf = tf + df, nr = tr + dr;
        if (on_board(nf, nr) && ((knights & occ_black) & (1ULL << sq_idx(nf, nr)))) return true;
    }
    for (auto [df, dr] : KING_DELTAS) {
        const int nf = tf + df, nr = tr + dr;
        if (on_board(nf, nr) && ((kings & occ_black) & (1ULL << sq_idx(nf, nr)))) return true;
    }
    // Black pawns capture toward decreasing rank: a black pawn on (tf±1, tr-1) attacks.
    for (int df : {-1, 1}) {
        const int nf = tf + df, nr = tr - 1;
        if (on_board(nf, nr) && ((pawns & occ_black) & (1ULL << sq_idx(nf, nr)))) return true;
    }
    const uint64_t bq = (bishops | queens) & occ_black;
    for (auto [df, dr] : BISHOP_DIRS) {
        int nf = tf + df, nr = tr + dr;
        while (on_board(nf, nr)) {
            const uint64_t m = 1ULL << sq_idx(nf, nr);
            if (occupied & m) {
                if (bq & m) return true;
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    const uint64_t rq = (rooks | queens) & occ_black;
    for (auto [df, dr] : ROOK_DIRS) {
        int nf = tf + df, nr = tr + dr;
        while (on_board(nf, nr)) {
            const uint64_t m = 1ULL << sq_idx(nf, nr);
            if (occupied & m) {
                if (rq & m) return true;
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    return false;
}

inline bool mask_attacked_by_black_fide(const WhiteBoard& b, uint64_t path, uint64_t occupied,
                                        uint64_t occ_black) {
    while (path) {
        const int sq = __builtin_ctzll(path);
        path &= path - 1;
        if (black_attacks_square_fide(occupied, b.pawns, b.knights, b.bishops, b.rooks, b.queens,
                                      b.kings, sq, occ_black))
            return true;
    }
    return false;
}

inline uint64_t between_mask(int a, int b) {
    const int lo = std::min(a, b), hi = std::max(a, b);
    uint64_t m = 0;
    for (int s = lo + 1; s < hi; ++s) m |= 1ULL << s;
    return m;
}

inline void white_castling_pseudo(const WhiteBoard& b, std::vector<WCandidate>& out) {
    const uint64_t own = b.occupied_white;
    const uint64_t king_bb = b.kings & own & WB_RANK_1;
    const int e1 = 4;
    if (king_bb == 0 || !(king_bb & (1ULL << e1))) return;  // non-chess960: king on e1
    const int king = e1;
    uint64_t candidates = b.castling_rights & b.rooks & own & WB_RANK_1;
    candidates &= (1ULL << 0) | (1ULL << 7);  // A1 | H1
    const uint64_t occ_black = b.occupied_black();
    while (candidates) {
        const int rook = __builtin_ctzll(candidates);
        candidates &= candidates - 1;
        const bool a_side = rook < king;
        const int king_to = a_side ? 2 : 6, rook_to = a_side ? 3 : 5;  // c1/d1 or g1/f1
        const uint64_t king_path = between_mask(king, king_to);
        const uint64_t rook_path = between_mask(rook, rook_to);
        const uint64_t kingm = 1ULL << king, rookm = 1ULL << rook;
        const uint64_t to_kingm = 1ULL << king_to, to_rookm = 1ULL << rook_to;
        if ((b.occupied ^ kingm ^ rookm) & (king_path | rook_path | to_kingm | to_rookm)) continue;
        if (mask_attacked_by_black_fide(b, king_path | kingm, b.occupied ^ kingm, occ_black)) continue;
        if (mask_attacked_by_black_fide(b, to_kingm, b.occupied ^ kingm ^ rookm ^ to_rookm, occ_black))
            continue;
        out.push_back(WCandidate{king, king_to, WPiece::King, std::nullopt, -1, false, true, !a_side,
                                 rook});
    }
}

inline std::vector<WCandidate> white_pseudo_legal(const WhiteBoard& b) {
    std::vector<WCandidate> out;
    const uint64_t occ = b.occupied, own = b.occupied_white, enemy = b.occupied_black();

    auto push_slider = [&](int from, WPiece piece, const std::pair<int, int>* dirs, int nd) {
        const int ff = from & 7, fr = from >> 3;
        for (int i = 0; i < nd; ++i) {
            const int df = dirs[i].first, dr = dirs[i].second;
            int nf = ff + df, nr = fr + dr;
            while (on_board(nf, nr)) {
                const int s = sq_idx(nf, nr);
                const uint64_t m = 1ULL << s;
                if (own & m) break;
                const int cap = (enemy & m) ? s : -1;
                out.push_back(WCandidate{from, s, piece, std::nullopt, cap, false, false, false, 0});
                if (occ & m) break;
                nf += df;
                nr += dr;
            }
        }
    };
    auto push_step = [&](int from, WPiece piece, const std::pair<int, int>* deltas, int nd) {
        const int ff = from & 7, fr = from >> 3;
        for (int i = 0; i < nd; ++i) {
            const int nf = ff + deltas[i].first, nr = fr + deltas[i].second;
            if (!on_board(nf, nr)) continue;
            const int s = sq_idx(nf, nr);
            const uint64_t m = 1ULL << s;
            if (own & m) continue;
            const int cap = (enemy & m) ? s : -1;
            out.push_back(WCandidate{from, s, piece, std::nullopt, cap, false, false, false, 0});
        }
    };

    uint64_t bbw = (b.knights & own) | (b.bishops & own) | (b.rooks & own) | (b.queens & own) |
                   (b.kings & own);
    while (bbw) {
        const int from = __builtin_ctzll(bbw);
        bbw &= bbw - 1;
        const uint64_t m = 1ULL << from;
        if (b.knights & m) push_step(from, WPiece::Knight, KNIGHT_DELTAS, 8);
        else if (b.kings & m) push_step(from, WPiece::King, KING_DELTAS, 8);
        else if (b.queens & m) {
            push_slider(from, WPiece::Queen, BISHOP_DIRS, 4);
            push_slider(from, WPiece::Queen, ROOK_DIRS, 4);
        } else if (b.bishops & m) push_slider(from, WPiece::Bishop, BISHOP_DIRS, 4);
        else if (b.rooks & m) push_slider(from, WPiece::Rook, ROOK_DIRS, 4);
    }

    white_castling_pseudo(b, out);

    const uint64_t pawns_w = b.pawns & own;
    static const char* PROMOS[4] = {"queen", "rook", "bishop", "knight"};
    // Captures (incl. promotions).
    uint64_t pw = pawns_w;
    while (pw) {
        const int from = __builtin_ctzll(pw);
        pw &= pw - 1;
        const int ff = from & 7, fr = from >> 3;
        for (int df : {-1, 1}) {
            const int nf = ff + df, nr = fr + 1;
            if (!on_board(nf, nr)) continue;
            const int s = sq_idx(nf, nr);
            if (!(enemy & (1ULL << s))) continue;
            if (nr == 7)
                for (const char* p : PROMOS)
                    out.push_back(
                        WCandidate{from, s, WPiece::Pawn, std::string(p), s, false, false, false, 0});
            else
                out.push_back(
                    WCandidate{from, s, WPiece::Pawn, std::nullopt, s, false, false, false, 0});
        }
    }
    // Single + double pushes (incl. promotions).
    const uint64_t single = (pawns_w << 8) & ~occ;
    const uint64_t dbl = (single << 8) & ~occ & WB_RANK_4;
    uint64_t sm = single;
    while (sm) {
        const int to = __builtin_ctzll(sm);
        sm &= sm - 1;
        const int from = to - 8;
        if ((to >> 3) == 7)
            for (const char* p : PROMOS)
                out.push_back(
                    WCandidate{from, to, WPiece::Pawn, std::string(p), -1, false, false, false, 0});
        else
            out.push_back(
                WCandidate{from, to, WPiece::Pawn, std::nullopt, -1, false, false, false, 0});
    }
    uint64_t dm = dbl;
    while (dm) {
        const int to = __builtin_ctzll(dm);
        dm &= dm - 1;
        out.push_back(
            WCandidate{to - 16, to, WPiece::Pawn, std::nullopt, -1, false, false, false, 0});
    }
    // En passant.
    if (b.ep_square >= 0) {
        const int ep = (int)b.ep_square;
        if (!(occ & (1ULL << ep))) {
            const int ef = ep & 7, er = ep >> 3;
            for (int df : {-1, 1}) {
                const int cf = ef + df, cr = er - 1;
                if (!on_board(cf, cr) || cr != 4) continue;
                const int cs = sq_idx(cf, cr);
                if (!(pawns_w & (1ULL << cs))) continue;
                out.push_back(WCandidate{cs, ep, WPiece::Pawn, std::nullopt, sq_idx(ef, er - 1), true,
                                         false, false, 0});
            }
        }
    }
    return out;
}

struct WhiteApply {
    uint64_t occupied, occupied_white;
    int white_king;  // -1 == none
    std::map<uint8_t, std::string> stacks;
};

inline WhiteApply apply_white_candidate(const WhiteBoard& b, const WCandidate& c,
                                        const std::map<uint8_t, std::string>& stacks) {
    const uint64_t from_m = 1ULL << c.from_sq, to_m = 1ULL << c.to_sq;
    uint64_t occ = b.occupied, own = b.occupied_white;
    int white_king = b.white_king_sq();
    std::map<uint8_t, std::string> new_stacks = stacks;
    if (c.capture_sq >= 0) {
        const uint64_t cap_m = 1ULL << c.capture_sq;
        occ &= ~cap_m;
        own &= ~cap_m;
        new_stacks.erase((uint8_t)c.capture_sq);
    }
    occ &= ~from_m;
    own &= ~from_m;
    occ |= to_m;
    own |= to_m;
    if (c.piece == WPiece::King) white_king = c.to_sq;
    if (c.is_castling) {
        const int rook_to = c.castling_kingside ? 5 : 3;  // f1 / d1
        const uint64_t rm = 1ULL << c.castling_rook_sq, rtm = 1ULL << rook_to;
        occ &= ~rm;
        own &= ~rm;
        occ |= rtm;
        own |= rtm;
    }
    return {occ, own, white_king, std::move(new_stacks)};
}

inline std::vector<WCandidate> white_legal_moves(const WhiteBoard& b,
                                                 const std::map<uint8_t, std::string>& stacks) {
    std::vector<WCandidate> out;
    for (auto& c : white_pseudo_legal(b)) {
        const auto ap = apply_white_candidate(b, c, stacks);
        if (white_in_chessckers_check(ap.occupied, ap.occupied_white, ap.white_king, ap.stacks))
            continue;
        if (c.is_castling) {
            // King's crossed squares (origin + intermediate) must not be attacked
            // under the Chessckers attack model.
            const int cross1 = c.castling_kingside ? 5 : 3;
            if (square_attacked_by_black_chessckers(b.occupied, b.occupied_white, stacks, 4) ||
                square_attacked_by_black_chessckers(b.occupied, b.occupied_white, stacks, cross1))
                continue;
        }
        out.push_back(c);
    }
    return out;
}

inline std::string white_uci(const WCandidate& c) {
    std::string s = square_name(c.from_sq) + square_name(c.to_sq);
    if (c.promotion) {
        const std::string& p = *c.promotion;
        s += (p == "queen") ? 'q' : (p == "rook") ? 'r' : (p == "bishop") ? 'b' : (p == "knight") ? 'n' : '?';
    }
    return s;
}

}  // namespace cc
