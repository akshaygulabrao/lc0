// Chessckers C++ engine — Phase 2 (lc0-split migration): pure-C++ move plumbing.
//
// A NativeMove is the GIL-free move representation for the threaded self-play
// loop. It carries the original generator output (an AnyMove for Black or a
// WCandidate for White) plus its uci; the apply payload (BlackMove/WhiteMove)
// and the NN move-feature encoding are DERIVED on demand. Nothing here touches
// pybind/py:: — every field maps byte-for-byte to what the Phase-1 dict path
// produced (gen_legal_dicts -> parse_*_move / encode_move_*_dict), so a pure
// game is identical to play_game_native. The final dict reconstruction (for the
// Python-visible training records) reuses the original `*_src` under the GIL.
#pragma once

#include <string>
#include <variant>
#include <vector>

#include "apply.hpp"      // BlackMove / WhiteMove + apply_black_move / apply_white_move
#include "encode.hpp"     // encode_move / encode_move_v2
#include "movegen.hpp"    // AnyMove + all_black_legal_moves
#include "movegen_white.hpp"  // WCandidate / white_legal_moves / wpiece_from_name / white_uci
#include "nn.hpp"         // ChesskersNet (is_v2 selects the encoder)

namespace cc {

// One legal move, kept in its native form. `black_src` is valid iff !is_white;
// `white_src` iff is_white. `is_castling_alt` marks the king-to-rook notation
// form (e1h1/e1a1) — same apply as the primary (e1g1/e1c1), different uci/encode.
struct NativeMove {
    std::string uci;
    bool is_white = false;
    bool is_castling_alt = false;
    AnyMove black_src;        // default = QuietMove; meaningful iff !is_white
    WCandidate white_src{};   // meaningful iff is_white
};

// -------- Black: AnyMove -> apply payload (mirrors parse_black_move) --------
inline BlackMove black_apply_of(const AnyMove& mv) {
    BlackMove bm;
    std::visit(
        [&](auto&& x) {
            using T = std::decay_t<decltype(x)>;
            bm.from_sq = parse_square(x.from_name);
            bm.to_sq = parse_square(x.to_name);
            if constexpr (std::is_same_v<T, DeployMove>) {
                bm.has_deploy_count = true;
                bm.deploy_count = x.deploy_count;
            } else if constexpr (std::is_same_v<T, ChargeMove>) {
                bm.has_capture = x.capture.has_value();
                bm.has_waypoints = x.waypoints.has_value();
                if (x.demoted_kings) bm.demoted_kings = *x.demoted_kings;
            } else if constexpr (std::is_same_v<T, ChainMove>) {
                bm.has_chain_hops = true;
                bm.has_capture = x.capture.has_value();
                bm.has_waypoints = x.waypoints.has_value();
                bm.chain_all_captures = x.chain_all_captures;
                bm.is_suicide = x.is_suicide;
                bm.chain_promotes = x.chain_promotes;
            }
            // QuietMove: all flags stay default (false / empty).
        },
        mv);
    return bm;
}

// -------- Black: AnyMove -> NN move features (mirrors encode_move_*_dict) -----
inline std::vector<float> black_encode(const ChesskersNet& net, const AnyMove& mv) {
    int from_sq = 0, to_sq = 0, deploy_count = 0, dem_req = 0;
    bool has_capture = false, has_deploy = false, has_dem = false;
    std::vector<std::string> wps;
    std::visit(
        [&](auto&& x) {
            using T = std::decay_t<decltype(x)>;
            from_sq = parse_square(x.from_name);
            to_sq = parse_square(x.to_name);
            if constexpr (std::is_same_v<T, DeployMove>) {
                has_deploy = true;
                deploy_count = x.deploy_count;
            } else if constexpr (std::is_same_v<T, ChargeMove>) {
                has_capture = x.capture.has_value();
                if (x.waypoints) wps = *x.waypoints;
                if (x.demotions_required) {
                    has_dem = true;
                    dem_req = *x.demotions_required;
                }
            } else if constexpr (std::is_same_v<T, ChainMove>) {
                has_capture = x.capture.has_value();
                if (x.waypoints) wps = *x.waypoints;
            }
        },
        mv);
    // Black moves never set a promotion key (promo word "" -> promo index 0).
    if (net.is_v2)
        return encode_move_v2(from_sq, to_sq, wps, has_capture, has_deploy, deploy_count, has_dem,
                              dem_req, "");
    return encode_move(from_sq, to_sq, has_capture, wps, has_deploy, deploy_count, has_dem, dem_req,
                       "");
}

// -------- White: WCandidate -> apply payload (mirrors parse_white_move) -------
inline WhiteMove white_apply_of(const WCandidate& c) {
    WhiteMove mv;
    mv.from_sq = c.from_sq;
    mv.to_sq = c.to_sq;
    mv.piece = c.piece;
    mv.has_promotion = c.promotion.has_value();
    if (mv.has_promotion) mv.promotion = wpiece_from_name(*c.promotion);
    mv.capture_sq = c.capture_sq;  // -1 == none
    if (c.is_castling) {
        mv.is_castling = true;
        mv.castling_kingside = c.castling_kingside;
        mv.castling_rook_sq = c.castling_rook_sq;
        mv.capture_sq = -1;
        mv.to_sq = c.castling_kingside ? 6 : 2;  // king destination (g1/c1)
    }
    return mv;
}

// -------- White: WCandidate -> NN move features (mirrors encode_move_*_dict) --
// `alt` selects the king-to-rook form: its dict "to" is the rook square.
inline std::vector<float> white_encode(const ChesskersNet& net, const WCandidate& c, bool alt) {
    const int from_sq = c.from_sq;
    const int to_sq = alt ? c.castling_rook_sq : c.to_sq;
    const bool has_capture = c.capture_sq >= 0;
    const std::string promo = c.promotion.value_or("");  // full word -> promo index 0
    if (net.is_v2)
        return encode_move_v2(from_sq, to_sq, {}, has_capture, false, 0, false, 0, promo);
    return encode_move(from_sq, to_sq, has_capture, {}, false, 0, false, 0, promo);
}

// -------- NativeMove dispatch (apply / encode) --------
inline Board apply_native(Board b, const NativeMove& m) {
    if (m.is_white) apply_white_move(b, white_apply_of(m.white_src));
    else apply_black_move(b, black_apply_of(m.black_src));
    return b;
}

inline std::vector<float> encode_native_move(const ChesskersNet& net, const NativeMove& m) {
    if (m.is_white) return white_encode(net, m.white_src, m.is_castling_alt);
    return black_encode(net, m.black_src);
}

// -------- Legal-move generation as NativeMoves --------
// Same content + order as gen_legal_dicts (both White castling forms emitted, the
// alt right after the primary).
inline std::vector<NativeMove> gen_legal_native(const Board& b) {
    std::vector<NativeMove> out;
    if (b.turn_white) {
        const WhiteBoard wb{b.occupied(),      b.occupied_white, b.pawns,  b.knights,
                            b.bishops,         b.rooks,          b.queens, b.kings,
                            b.castling_rights, (long)b.ep_square};
        for (const auto& c : white_legal_moves(wb, b.stacks)) {
            NativeMove nm;
            nm.is_white = true;
            nm.white_src = c;
            nm.uci = white_uci(c);
            out.push_back(std::move(nm));
            if (c.is_castling) {
                NativeMove alt;
                alt.is_white = true;
                alt.white_src = c;
                alt.is_castling_alt = true;
                alt.uci = square_name(c.from_sq) + square_name(c.castling_rook_sq);
                out.push_back(std::move(alt));
            }
        }
        return out;
    }
    const uint64_t wk = b.kings & b.occupied_white;
    const long king_sq = wk ? __builtin_ctzll(wk) : -1;
    for (const auto& mv : all_black_legal_moves(b.occupied(), b.occupied_white, king_sq, b.stacks)) {
        NativeMove nm;
        nm.is_white = false;
        nm.black_src = mv;
        std::visit([&](auto&& x) { nm.uci = x.uci; }, mv);
        out.push_back(std::move(nm));
    }
    return out;
}

}  // namespace cc
