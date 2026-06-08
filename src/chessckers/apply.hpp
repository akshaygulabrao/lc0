// Chessckers C++ engine — Slice 5a: apply-move + status.
//
// State advancement for the search tree. Black apply is a 1:1 port of PyVariant's
// _apply_* helpers (manual bitboard/stack mutation; turn flips to White; castling
// rights, ep and clocks are untouched — Black is a White-only-castling variant
// and PyVariant never pushes a Black move through python-chess). White apply
// (the python-chess board.push port) lands in a follow-up.
//
// detect_status mirrors client._detect_status + the move-gen-derived terminal
// states in _state_to_dict / status_and_legal.
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "board.hpp"
#include "movegen.hpp"
#include "movegen_white.hpp"

namespace cc {

// -------- Board mutation primitives --------

inline void bb_remove_piece(Board& b, int sq) {
    const uint64_t m = ~(1ULL << sq);
    b.pawns &= m;
    b.knights &= m;
    b.bishops &= m;
    b.rooks &= m;
    b.queens &= m;
    b.kings &= m;
    b.occupied_white &= m;
    b.occupied_black &= m;
}

inline void bb_set_black_pawn(Board& b, int sq) {
    bb_remove_piece(b, sq);
    const uint64_t m = 1ULL << sq;
    b.pawns |= m;
    b.occupied_black |= m;
}

inline void bb_set_black_king(Board& b, int sq) {
    bb_remove_piece(b, sq);
    const uint64_t m = 1ULL << sq;
    b.kings |= m;
    b.occupied_black |= m;
}

// Sync the bitboard piece at sq with a stack's top char (k -> black king,
// s/S -> black pawn). Mirrors _set_top_piece_on_board.
inline void set_top_piece_on_board(Board& b, int sq, char top) {
    if (top == 'k') bb_set_black_king(b, sq);
    else if (top == 's' || top == 'S') bb_set_black_pawn(b, sq);
    else bb_remove_piece(b, sq);
}

// Move the whole tower from->to, merging onto a friendly destination (incoming
// on top). `override_pieces` substitutes the moving stack (sprint / demotion).
inline void move_full_tower(Board& b, int from_sq, int to_sq,
                            const std::string* override_pieces = nullptr) {
    const std::string moving = override_pieces ? *override_pieces : b.stacks.at((uint8_t)from_sq);
    b.stacks.erase((uint8_t)from_sq);
    bb_remove_piece(b, from_sq);
    std::string existing;
    const auto it = b.stacks.find((uint8_t)to_sq);
    if (it != b.stacks.end()) existing = it->second;
    const std::string new_stack = existing + moving;  // incoming on top
    b.stacks[(uint8_t)to_sq] = new_stack;
    set_top_piece_on_board(b, to_sq, new_stack.back());
}

inline bool is_orthogonal(int from, int to) {
    return ((from & 7) == (to & 7) || (from >> 3) == (to >> 3)) && from != to;
}

// -------- Black move apply --------
//
// The fields the PyVariant apply path reads off a move dict, extracted once.
struct BlackMove {
    int from_sq, to_sq;
    bool has_deploy_count = false;
    int deploy_count = 0;
    bool has_chain_hops = false;
    bool has_capture = false;
    bool has_waypoints = false;  // overshoot charge
    std::vector<std::string> chain_all_captures;
    bool is_suicide = false;
    bool chain_promotes = false;
    std::vector<int> demoted_kings;  // empty -> forced (all kings)
};

inline void apply_quiet_or_sprint(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const bool is_sprint = pieces == "s" && (mv.from_sq >> 3) == 7 &&
                           std::abs((mv.to_sq >> 3) - (mv.from_sq >> 3)) == 2;
    if (is_sprint) {
        const std::string ov = "S";
        move_full_tower(b, mv.from_sq, mv.to_sq, &ov);
    } else {
        move_full_tower(b, mv.from_sq, mv.to_sq);
    }
    if ((mv.to_sq >> 3) == 0) {  // rank-1 promotion (after the merge)
        const std::string promoted = promote_all_stones(b.stacks.at((uint8_t)mv.to_sq));
        b.stacks[(uint8_t)mv.to_sq] = promoted;
        set_top_piece_on_board(b, mv.to_sq, promoted.back());
    }
}

inline void apply_deploy(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const int s = mv.deploy_count;
    const std::string sub = pieces.substr(pieces.size() - s);
    const std::string remainder = pieces.substr(0, pieces.size() - s);
    b.stacks[(uint8_t)mv.from_sq] = remainder;
    set_top_piece_on_board(b, mv.from_sq, remainder.back());
    std::string existing;
    const auto it = b.stacks.find((uint8_t)mv.to_sq);
    if (it != b.stacks.end()) existing = it->second;
    std::string new_stack = existing + sub;
    if ((mv.to_sq >> 3) == 0) new_stack = promote_all_stones(new_stack);
    b.stacks[(uint8_t)mv.to_sq] = new_stack;
    set_top_piece_on_board(b, mv.to_sq, new_stack.back());
}

inline void apply_charge(Board& b, const BlackMove& mv) {
    const std::string pieces = b.stacks.at((uint8_t)mv.from_sq);
    const int ff = mv.from_sq & 7, fr = mv.from_sq >> 3, tf = mv.to_sq & 7, tr = mv.to_sq >> 3;
    const int df = tf - ff, dr = tr - fr;
    const int d = std::max(std::abs(df), std::abs(dr));
    const int dfs = d ? df / d : 0, drs = d ? dr / d : 0;
    const bool is_rim_overshoot = mv.has_waypoints;
    const int last_step = is_rim_overshoot ? d : d - 1;
    for (int k = 1; k <= last_step; ++k) {
        const int sq = sq_idx(ff + k * dfs, fr + k * drs);
        if (owner(b.occupied(), b.occupied_white, sq) == SQ_WHITE) bb_remove_piece(b, sq);
    }
    if (!is_rim_overshoot && owner(b.occupied(), b.occupied_white, mv.to_sq) == SQ_WHITE) {
        b.stacks.erase((uint8_t)mv.from_sq);  // ram: tower destroyed, landing White stays
        bb_remove_piece(b, mv.from_sq);
        return;
    }
    std::vector<int> king_positions;
    for (int i = 0; i < (int)pieces.size(); ++i)
        if (pieces[i] == 'k') king_positions.push_back(i + 1);
    const std::vector<int>& chosen = mv.demoted_kings.empty() ? king_positions : mv.demoted_kings;
    std::string new_pieces = pieces;
    for (int pos : chosen) new_pieces[pos - 1] = 'S';
    move_full_tower(b, mv.from_sq, mv.to_sq, &new_pieces);
}

inline void apply_diagonal_capture(Board& b, const BlackMove& mv) {
    const int ff = mv.from_sq & 7, fr = mv.from_sq >> 3, tf = mv.to_sq & 7, tr = mv.to_sq >> 3;
    const int df = tf - ff, dr = tr - fr;
    const int d = std::max(std::abs(df), std::abs(dr));
    const int dfs = d ? df / d : 0, drs = d ? dr / d : 0;
    for (int k = 1; k < d; ++k) {
        const int sq = sq_idx(ff + k * dfs, fr + k * drs);
        if (owner(b.occupied(), b.occupied_white, sq) == SQ_WHITE) bb_remove_piece(b, sq);
    }
    if (owner(b.occupied(), b.occupied_white, mv.to_sq) == SQ_WHITE) {
        b.stacks.erase((uint8_t)mv.from_sq);
        bb_remove_piece(b, mv.from_sq);
        return;
    }
    move_full_tower(b, mv.from_sq, mv.to_sq);
}

inline void apply_chain_move(Board& b, const BlackMove& mv) {
    const std::string orig_stack = b.stacks.at((uint8_t)mv.from_sq);
    for (const std::string& cap : mv.chain_all_captures) bb_remove_piece(b, parse_square(cap));
    b.stacks.erase((uint8_t)mv.from_sq);
    bb_remove_piece(b, mv.from_sq);
    if (mv.is_suicide) return;
    const std::string final_stack = mv.chain_promotes ? promote_all_stones(orig_stack) : orig_stack;
    b.stacks[(uint8_t)mv.to_sq] = final_stack;
    if (final_stack.back() == 'k') bb_set_black_king(b, mv.to_sq);
    else bb_set_black_pawn(b, mv.to_sq);
}

// Dispatch mirrors apply_black_move_known. Mutates b in place; flips turn to
// White. castling/ep/clocks are deliberately left unchanged.
inline void apply_black_move(Board& b, const BlackMove& mv) {
    if (mv.has_deploy_count) apply_deploy(b, mv);
    else if (is_orthogonal(mv.from_sq, mv.to_sq)) apply_charge(b, mv);
    else if (mv.has_chain_hops) apply_chain_move(b, mv);
    else if (mv.has_capture) apply_diagonal_capture(b, mv);
    else apply_quiet_or_sprint(b, mv);
    b.turn_white = true;
    // Rank-8 counter (#3): a check from Black at any point resets it to 0.
    if (b.rank8_count != 0) {
        const uint64_t wk_bb = b.kings & b.occupied_white;
        if (wk_bb != 0 &&
            white_in_chessckers_check(b.occupied(), b.occupied_white, __builtin_ctzll(wk_bb), b.stacks))
            b.rank8_count = 0;
    }
}

// -------- White move apply (python-chess board.push port) --------

// python-chess Board.clean_castling_rights (non-chess960): keep a right only if
// a friendly rook sits on that corner AND the king is on its home square. This
// drops a right whose rook was captured by a (manual) Black move before White's
// next push observes it.
inline uint64_t clean_castling_rights(const Board& b) {
    const uint64_t castling = b.castling_rights & b.rooks;
    uint64_t white = castling & BB_RANK_1 & b.occupied_white & (BB_A1 | BB_H1);
    uint64_t black = castling & BB_RANK_8 & b.occupied_black & (BB_A8 | BB_H8);
    if (!(b.occupied_white & b.kings & (1ULL << 4))) white = 0;    // white king must be on e1
    if (!(b.occupied_black & b.kings & (1ULL << 60))) black = 0;   // black king must be on e8
    return white | black;
}

inline void set_white_piece(Board& b, int sq, WPiece t) {
    bb_remove_piece(b, sq);
    const uint64_t m = 1ULL << sq;
    switch (t) {
        case WPiece::Pawn: b.pawns |= m; break;
        case WPiece::Knight: b.knights |= m; break;
        case WPiece::Bishop: b.bishops |= m; break;
        case WPiece::Rook: b.rooks |= m; break;
        case WPiece::Queen: b.queens |= m; break;
        case WPiece::King: b.kings |= m; break;
    }
    b.occupied_white |= m;
}

// Castling is given with to_sq = the king's DESTINATION (g1/c1); castling_rook_sq
// is the rook origin. (Both notation forms e1g1 and e1h1 reduce to this.)
struct WhiteMove {
    int from_sq, to_sq;
    WPiece piece;
    bool has_promotion = false;
    WPiece promotion = WPiece::Queen;
    int capture_sq = -1;  // captured square (whole Black tower removed); -1 == none
    bool is_castling = false;
    bool castling_kingside = false;
    int castling_rook_sq = 0;
};

inline void apply_white_move(Board& b, const WhiteMove& mv) {
    const int from = mv.from_sq, to = mv.to_sq;
    // Castling rights: clean on the parent state, then clear from/to bits; a
    // king move clears all White back-rank rights.
    uint64_t cr = clean_castling_rights(b);
    cr &= ~(1ULL << from) & ~(1ULL << to);
    if (mv.piece == WPiece::King) cr &= ~BB_RANK_1;
    // En passant target: only on a White double push.
    const int new_ep = (mv.piece == WPiece::Pawn && (to - from) == 16 && (from >> 3) == 1)
                           ? from + 8
                           : -1;
    // Remove captured square (the whole Black tower overlay too).
    if (mv.capture_sq >= 0) {
        bb_remove_piece(b, mv.capture_sq);
        b.stacks.erase((uint8_t)mv.capture_sq);
    }
    bb_remove_piece(b, from);
    if (mv.is_castling) {
        bb_remove_piece(b, mv.castling_rook_sq);
        set_white_piece(b, mv.castling_kingside ? 6 : 2, WPiece::King);  // g1/c1
        set_white_piece(b, mv.castling_kingside ? 5 : 3, WPiece::Rook);  // f1/d1
    } else {
        set_white_piece(b, to, mv.has_promotion ? mv.promotion : mv.piece);
    }
    b.castling_rights = cr;
    b.ep_square = new_ep;
    b.halfmove = (mv.capture_sq >= 0 || mv.piece == WPiece::Pawn) ? 0 : b.halfmove + 1;
    // Opening double-move (#2): the first of White's two sub-moves keeps the turn
    // White; otherwise the turn passes to Black and the rank-8 counter (#3) is
    // updated — tick if the king ended on rank 8 (it can't be in check after a
    // legal White move), else reset.
    if (b.white_moves_left >= 2) {
        b.white_moves_left -= 1;
    } else {
        b.white_moves_left = 1;
        b.turn_white = false;
        const uint64_t wk_bb = b.kings & b.occupied_white;
        if (wk_bb != 0 && (__builtin_ctzll(wk_bb) >> 3) == 7) b.rank8_count += 1;
        else b.rank8_count = 0;
    }
}

// -------- Status detection --------

struct Status {
    std::string status;  // "" == None
    std::string winner;  // "" == None
};

inline Status detect_status(const Board& b) {
    if (b.stacks.empty()) return {"variantEnd", "white"};  // Black eliminated
    if (b.rank8_count >= 3) return {"variantEnd", "white"};  // rank-8 hold (#3)
    const uint64_t wk_bb = b.kings & b.occupied_white;
    if (wk_bb == 0) return {"variantEnd", "black"};  // White king captured
    const int wk_sq = __builtin_ctzll(wk_bb);
    if (b.turn_white) {
        const WhiteBoard wb{b.occupied(),  b.occupied_white, b.pawns,           b.knights,
                            b.bishops,     b.rooks,          b.queens,          b.kings,
                            b.castling_rights, (long)b.ep_square};
        if (white_legal_moves(wb, b.stacks).empty()) {
            const bool check =
                white_in_chessckers_check(b.occupied(), b.occupied_white, wk_sq, b.stacks);
            return check ? Status{"mate", "black"} : Status{"stalemate", ""};
        }
    } else {
        if (all_black_legal_moves(b.occupied(), b.occupied_white, wk_sq, b.stacks).empty())
            return {"variantEnd", "white"};
    }
    return {"", ""};
}

}  // namespace cc
