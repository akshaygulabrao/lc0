// Chessckers C++ engine — Slice 1: the §3B capture atom.
//
// 1:1 port of the validated Rust `find_capture_hops` (rust/chessckers_movegen/
// src/lib.rs) and the pure-Python reference `_find_capture_hops`
// (variant_py/moves_black.py). Walks one straight diagonal from a tower square
// and emits a CaptureHop for every legal landing along it (normal landings,
// k>d rams, rim-T landings, and the off-grid overshoot). Held byte-equivalent
// to the Python via tests/test_cpp_capture_hops.py (exhaustive square×dir×n).
//
// No bouncing (spec §3B step 3): a straight diagonal that would leave the 10×10
// grid simply terminates. The 10×10 grid = the 8×8 board plus a one-square rim
// ring; rim squares carry no pieces and have square index -1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "board.hpp"  // cc::square_name

namespace cc {

// -------- Geometry --------

constexpr int MAX_HOP_STEPS = 26;

constexpr int SQ_EMPTY = 0;
constexpr int SQ_WHITE = 1;
constexpr int SQ_BLACK = 2;

inline constexpr std::pair<int, int> ORTHO_DIRS[4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

inline bool on_board(int f, int r) { return f >= 0 && f <= 7 && r >= 0 && r <= 7; }
inline bool on_grid(int f, int r) { return f >= -1 && f <= 8 && r >= -1 && r <= 8; }
inline int sq_idx(int f, int r) { return (r << 3) | f; }

// (file, rank) on the 10×10 grid packed as a single byte: rank10*10 + file10,
// where file10/rank10 = board coord + 1 (rim -1 -> 0, rim 8 -> 9). Range 0..99.
inline uint8_t coord10_of(int f, int r) { return (uint8_t)((r + 1) * 10 + (f + 1)); }
inline int c10_file(uint8_t c10) { return c10 % 10 - 1; }  // board file, -1..8
inline int c10_rank(uint8_t c10) { return c10 / 10 - 1; }  // board rank, -1..8

// coord10 -> the 2-char key string. Byte-identical to the historical coord_key
// (Python _COORD_KEY / Rust coord_key): rim files 'z'/'i' for -1/8, rank digit
// '0'..'9' for -1..8.
inline std::string key_str(uint8_t c10) {
    const int f10 = c10 % 10, r10 = c10 / 10;
    std::string s;
    s += (f10 == 0) ? 'z' : (f10 == 9) ? 'i' : char('a' + f10 - 1);
    s += char('0' + r10);
    return s;
}

inline int owner(uint64_t occupied, uint64_t occupied_white, int sq) {
    const uint64_t mask = 1ULL << sq;
    if (!(occupied & mask)) return SQ_EMPTY;
    if (occupied_white & mask) return SQ_WHITE;
    return SQ_BLACK;
}

// -------- Capture-path table (pure geometry, no board state) --------

struct PathStep {
    int f, r;
    int sq;  // -1 if rim
    uint8_t c10;
    int df, dr;
    bool did_bounce;  // always false (kept for shape parity with Rust/Python)
};

inline const std::map<std::tuple<int, int, int, int>, std::vector<PathStep>>& capture_paths() {
    static const auto table = [] {
        std::map<std::tuple<int, int, int, int>, std::vector<PathStep>> paths;
        for (int f0 = -1; f0 <= 8; ++f0)
            for (int r0 = -1; r0 <= 8; ++r0)
                for (int df0 : {-1, 1})
                    for (int dr0 : {-1, 1}) {
                        std::vector<PathStep> steps;
                        steps.reserve(MAX_HOP_STEPS);
                        int f = f0, r = r0;
                        for (int i = 0; i < MAX_HOP_STEPS; ++i) {
                            const int nf = f + df0, nr = r + dr0;
                            if (nf < -1 || nf > 8 || nr < -1 || nr > 8) break;
                            f = nf;
                            r = nr;
                            const int sq = on_board(f, r) ? sq_idx(f, r) : -1;
                            steps.push_back(PathStep{f, r, sq, coord10_of(f, r), df0, dr0, false});
                        }
                        paths[{f0, r0, df0, dr0}] = std::move(steps);
                    }
        return paths;
    }();
    return table;
}

// -------- CaptureHop + find_capture_hops --------

struct CaptureHop {
    int df, dr;                          // direction
    uint8_t landing_c10;                 // coord10 of the landing key
    int landing_square;                  // -1 == None (rim / overshoot)
    std::vector<int> captures;           // board squares of Whites captured on the path
    std::vector<uint8_t> waypoints;      // every traced step's coord10 (incl. landing)
    bool is_suicide;
    bool crossed_rank1;
    int cadence;                         // landing distance k
    bool is_overshoot;
};

// Walk (df0, dr0) up to n+1 steps from (f0, r0). Emits a CaptureHop for every
// legal landing. Ports the Python/Rust logic exactly, including:
//  - rams (is_suicide) require k>d: emitted only when captures already exist,
//    and BEFORE the landing White is added to captures (no double-count);
//  - a friendly Black tower terminates the trace (a block, not an off-grid exit
//    -> no overshoot past it);
//  - the off-grid overshoot is a candidate DISTINCT from a rim landing at the
//    same key (different cadence), so both are kept.
inline std::vector<CaptureHop> find_capture_hops(
    uint64_t occupied, uint64_t occupied_white,
    const StackMap& stacks,
    int f0, int r0, int df0, int dr0, int n) {
    std::vector<CaptureHop> options;
    std::vector<int> captures_so_far;
    uint64_t captured_set = 0;
    std::vector<uint8_t> waypoints_so_far;
    bool crossed_rank1 = false;
    bool friendly_blocked = false;

    const auto it = capture_paths().find({f0, r0, df0, dr0});
    if (it == capture_paths().end()) return options;
    const std::vector<PathStep>& path = it->second;
    const int max_step = n + 1;

    for (int step_idx = 0; step_idx < (int)path.size(); ++step_idx) {
        if (step_idx >= max_step) break;
        const PathStep& step = path[step_idx];
        const uint8_t cur_key = step.c10;
        waypoints_so_far.push_back(cur_key);
        if (step.r == 0) crossed_rank1 = true;
        const int step_num = step_idx + 1;  // 1-based landing distance k

        if (step.sq >= 0) {
            const int sq = step.sq;
            const uint64_t cap_mask = 1ULL << sq;
            if (captured_set & cap_mask) {
                // Revisit of an already-captured square (straight paths never
                // hit this, but kept for parity with the shared Python logic).
                if (!captures_so_far.empty())
                    options.push_back(CaptureHop{step.df, step.dr, cur_key, sq, captures_so_far,
                                                 waypoints_so_far, false, crossed_rank1, step_num, false});
            } else {
                const int o = owner(occupied, occupied_white, sq);
                if (o == SQ_EMPTY) {
                    if (!captures_so_far.empty())
                        options.push_back(CaptureHop{step.df, step.dr, cur_key, sq, captures_so_far,
                                                     waypoints_so_far, false, crossed_rank1, step_num, false});
                } else if (o == SQ_BLACK && stacks.count((uint8_t)sq)) {
                    friendly_blocked = true;
                    break;
                } else {
                    // White (or Black-without-stack, defensive). Ram before add.
                    if (!captures_so_far.empty())
                        options.push_back(CaptureHop{step.df, step.dr, cur_key, sq, captures_so_far,
                                                     waypoints_so_far, true, crossed_rank1, step_num, false});
                    captures_so_far.push_back(sq);
                    captured_set |= cap_mask;
                }
            }
        } else {
            // Rim square (T) — never friendly.
            if (!captures_so_far.empty())
                options.push_back(CaptureHop{step.df, step.dr, cur_key, -1, captures_so_far,
                                             waypoints_so_far, false, crossed_rank1, step_num, false});
        }
    }

    // §3B off-grid overshoot: path left the grid before the cadence limit, not
    // friendly-blocked, and captured >=1 White -> settles on the last on-board
    // square (resolved later) and ends the chain. cadence = path.len()+1.
    if (!friendly_blocked && !captures_so_far.empty() && (int)path.size() < max_step) {
        options.push_back(CaptureHop{df0, dr0, waypoints_so_far.back(), -1, captures_so_far,
                                     waypoints_so_far, false, crossed_rank1, (int)path.size() + 1, true});
    }
    return options;
}

// -------- Capture chains (Slice 2a) --------
//
// 1:1 port of the Rust chain machinery (next_capture_options, build_final_move,
// apply_hop, enumerate_chains[_recursive], first_hop_suicides,
// black_diagonal_capture_moves_native) and the pure-Python reference in
// moves_black.py. Builds the full capture MoveDicts on top of the hop atom.

inline std::vector<std::pair<int, int>> dirs_for_top(char top) {
    if (top == 'k') return {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    return {{-1, -1}, {1, -1}};
}

inline bool hop_promotes(const CaptureHop& hop) {
    if (hop.crossed_rank1) return true;
    if (hop.landing_square >= 0 && (hop.landing_square >> 3) == 0) return true;
    return false;
}

inline Tower promote_all_stones(const Tower& stack) {
    Tower out = stack;
    for (char& c : out)
        if (c == 's' || c == 'S') c = 'k';
    return out;
}

struct ChainMove {
    std::string uci;
    uint8_t from_sq = 0, to_sq = 0;          // 8x8 square indices
    bool is_king = false;                     // "king" vs "pawn"
    int16_t capture_sq = -1;                  // 8x8 square; -1 == none
    bool has_waypoints = false;               // null-vs-present (JSON parity)
    std::vector<uint8_t> waypoints;           // coord10
    std::vector<uint8_t> chain_hops;          // coord10
    std::vector<uint8_t> chain_all_captures;  // 8x8 squares
    bool is_suicide = false;
    bool chain_promotes = false;
    int cadence = 0;
};

// dedup + cadence-lock + last-dir(no-reversal) + suicide filter over the hop
// atom, across the directions valid for the current tower top.
inline std::vector<CaptureHop> next_capture_options(
    uint64_t occupied, uint64_t occupied_white, const StackMap& stacks,
    int cf, int cr, const Tower& cur_stack, bool has_last_dir, int ldf, int ldr, int n,
    bool has_cadence, int cadence, bool include_suicide) {
    std::vector<CaptureHop> options;
    if (cur_stack.empty()) return options;
    for (auto [df, dr] : dirs_for_top(cur_stack.back())) {
        if (has_last_dir && df == -ldf && dr == -ldr) continue;
        for (auto& hop : find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n))
            options.push_back(hop);
    }
    if (!include_suicide) {
        std::vector<CaptureHop> t;
        for (auto& h : options)
            if (!h.is_suicide) t.push_back(h);
        options = std::move(t);
    }
    if (has_cadence) {
        std::vector<CaptureHop> t;
        for (auto& h : options)
            if (h.cadence == cadence) t.push_back(h);
        options = std::move(t);
    }
    // identity = (df, dr, landing_c10, captures, is_suicide, is_overshoot, cadence)
    std::set<std::tuple<int, int, uint8_t, std::vector<int>, bool, bool, int>> seen;
    std::vector<CaptureHop> deduped;
    for (auto& h : options) {
        auto key = std::make_tuple(h.df, h.dr, h.landing_c10, h.captures, h.is_suicide,
                                   h.is_overshoot, h.cadence);
        if (seen.insert(key).second) deduped.push_back(h);
    }
    return deduped;
}

inline ChainMove build_final_move(int chain_start, const Tower& orig_stack,
                                  const std::vector<CaptureHop>& hops) {
    const bool is_suicide_chain = !hops.empty() && hops.back().is_suicide;
    std::vector<int> all_captures;
    std::vector<uint8_t> all_waypoints, hop_keys;
    for (auto& h : hops) {
        all_captures.insert(all_captures.end(), h.captures.begin(), h.captures.end());
        all_waypoints.insert(all_waypoints.end(), h.waypoints.begin(), h.waypoints.end());
        hop_keys.push_back(h.landing_c10);
    }

    const int last_landing = hops.empty() ? -1 : hops.back().landing_square;
    int final_landing;
    if (last_landing >= 0) {
        final_landing = last_landing;
    } else {
        // End-of-turn fallback: last on-board waypoint, else the chain start.
        final_landing = chain_start;
        for (auto it = all_waypoints.rbegin(); it != all_waypoints.rend(); ++it) {
            const int f = c10_file(*it), r = c10_rank(*it);
            if (on_board(f, r)) {
                final_landing = sq_idx(f, r);
                break;
            }
        }
    }

    char final_top;
    if (is_suicide_chain) {
        final_top = orig_stack.back();
    } else {
        Tower stack_thru = orig_stack;
        for (auto& h : hops)
            if (hop_promotes(h)) stack_thru = promote_all_stones(stack_thru);
        final_top = stack_thru.back();
    }

    int capture_sq = -1;
    if (!all_captures.empty()) capture_sq = all_captures[0];
    else if (is_suicide_chain) capture_sq = final_landing;

    const int cadence = hops[0].cadence;
    std::string uci = "c" + std::to_string(cadence) + ":" + square_name(chain_start);
    for (uint8_t hk : hop_keys) {
        uci += '~';
        uci += key_str(hk);
    }
    uci += "->";
    uci += square_name(final_landing);

    bool chain_promotes_any = false;
    for (auto& h : hops)
        if (hop_promotes(h)) {
            chain_promotes_any = true;
            break;
        }

    ChainMove m;
    m.uci = std::move(uci);
    m.from_sq = (uint8_t)chain_start;
    m.to_sq = (uint8_t)final_landing;
    m.is_king = (final_top == 'k');
    m.capture_sq = (int16_t)capture_sq;
    if (hops.size() > 1) {
        m.has_waypoints = true;
        m.waypoints = std::move(all_waypoints);
    }
    m.chain_hops = std::move(hop_keys);
    m.chain_all_captures.assign(all_captures.begin(), all_captures.end());
    m.is_suicide = is_suicide_chain;
    m.chain_promotes = chain_promotes_any;
    m.cadence = cadence;
    return m;
}

struct HopApply {
    uint64_t occupied, occupied_white;
    StackMap stacks;
    Tower land_stack;
};

inline HopApply apply_hop(uint64_t occupied, uint64_t occupied_white,
                          StackMap stacks, int cf, int cr,
                          const Tower& cur_stack, const CaptureHop& hop) {
    if (on_board(cf, cr)) {
        const int cur_sq = sq_idx(cf, cr);
        const uint64_t m = ~(1ULL << cur_sq);
        occupied &= m;
        occupied_white &= m;
        stacks.erase((uint8_t)cur_sq);
    }
    for (int cap_sq : hop.captures) {
        const uint64_t m = ~(1ULL << cap_sq);
        occupied &= m;
        occupied_white &= m;
    }
    Tower land_stack = hop_promotes(hop) ? promote_all_stones(cur_stack) : cur_stack;
    if (hop.landing_square >= 0) {
        const uint64_t mask = 1ULL << hop.landing_square;
        occupied |= mask;
        occupied_white &= ~mask;  // Black
        stacks[(uint8_t)hop.landing_square] = land_stack;
    }
    return {occupied, occupied_white, std::move(stacks), std::move(land_stack)};
}

inline void enumerate_chains_recursive(uint64_t occupied, uint64_t occupied_white, long king_sq,
                                       const StackMap& stacks, int chain_start,
                                       int cf, int cr, const Tower& cur_stack,
                                       bool has_last_dir, int ldf, int ldr,
                                       std::vector<CaptureHop> hops_so_far, bool has_cadence,
                                       int cadence, int n, const Tower& orig_stack,
                                       std::vector<ChainMove>& results) {
    // White-king-captured short-circuit (game over -> stop extending the chain).
    if (king_sq >= 0 && (occupied_white & (1ULL << king_sq)) == 0) return;
    auto options = next_capture_options(occupied, occupied_white, stacks, cf, cr, cur_stack,
                                        has_last_dir, ldf, ldr, n, has_cadence, cadence, false);
    if (options.empty()) return;
    for (auto& hop : options) {
        std::vector<CaptureHop> hops_next = hops_so_far;
        hops_next.push_back(hop);
        results.push_back(build_final_move(chain_start, orig_stack, hops_next));
        if (hop.is_overshoot) continue;
        auto ap = apply_hop(occupied, occupied_white, stacks, cf, cr, cur_stack, hop);
        int nf, nr;
        if (hop.landing_square >= 0) {
            nf = hop.landing_square & 7;
            nr = hop.landing_square >> 3;
        } else {
            nf = c10_file(hop.landing_c10);
            nr = c10_rank(hop.landing_c10);
        }
        const int next_cadence = has_cadence ? cadence : hop.cadence;
        enumerate_chains_recursive(ap.occupied, ap.occupied_white, king_sq, ap.stacks, chain_start,
                                   nf, nr, ap.land_stack, true, hop.df, hop.dr, hops_next, true,
                                   next_cadence, n, orig_stack, results);
    }
}

inline std::vector<ChainMove> enumerate_chains(uint64_t occupied, uint64_t occupied_white,
                                               long king_sq,
                                               const StackMap& stacks,
                                               int chain_start) {
    const auto it = stacks.find((uint8_t)chain_start);
    if (it == stacks.end() || it->second.empty()) return {};
    const Tower orig_stack = it->second;
    const int n = (int)orig_stack.size();
    std::vector<ChainMove> results;
    enumerate_chains_recursive(occupied, occupied_white, king_sq, stacks, chain_start,
                               chain_start & 7, chain_start >> 3, orig_stack, false, 0, 0, {}, false,
                               0, n, orig_stack, results);
    return results;
}

inline std::vector<ChainMove> first_hop_suicides(uint64_t occupied, uint64_t occupied_white,
                                                 const StackMap& stacks,
                                                 int chain_start) {
    const auto it = stacks.find((uint8_t)chain_start);
    if (it == stacks.end() || it->second.empty()) return {};
    const Tower pieces = it->second;
    const int n = (int)pieces.size();
    const int cf = chain_start & 7, cr = chain_start >> 3;
    std::vector<ChainMove> moves;
    for (auto [df, dr] : dirs_for_top(pieces.back()))
        for (auto& hop : find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n))
            if (hop.is_suicide) moves.push_back(build_final_move(chain_start, pieces, {hop}));
    return moves;
}

inline std::vector<ChainMove> black_diagonal_capture_moves(uint64_t occupied, uint64_t occupied_white,
                                                           long king_sq,
                                                           const StackMap& stacks) {
    std::vector<ChainMove> moves;
    // std::map iterates ascending key order -> matches Rust keys.sort_unstable().
    for (const auto& [sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        auto chains = enumerate_chains(occupied, occupied_white, king_sq, stacks, sq);
        moves.insert(moves.end(), chains.begin(), chains.end());
        auto suis = first_hop_suicides(occupied, occupied_white, stacks, sq);
        moves.insert(moves.end(), suis.begin(), suis.end());
    }
    return moves;
}

// -------- Quiet diagonals + sprint (Slice 2b) --------

struct QuietMove {
    std::string uci;
    uint8_t from_sq = 0, to_sq = 0;
    bool is_king = false;  // "king" vs "pawn"
};

inline QuietMove build_quiet(int from_sq, int to_sq, char top) {
    QuietMove m;
    m.uci = square_name(from_sq) + square_name(to_sq);
    m.from_sq = (uint8_t)from_sq;
    m.to_sq = (uint8_t)to_sq;
    m.is_king = (top == 'k');
    return m;
}

inline std::vector<QuietMove> black_diagonal_quiet_moves(uint64_t occupied, uint64_t occupied_white,
                                                         const StackMap& stacks) {
    std::vector<QuietMove> moves;
    for (const auto& [from_sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        const int height = (int)pieces.size();
        const char top = pieces.back();
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;

        for (auto [df, dr] : dirs_for_top(top)) {
            for (int k = 1; k <= height; ++k) {
                const int tf = from_file + k * df, tr = from_rank + k * dr;
                if (!on_board(tf, tr)) break;
                const int to_sq = sq_idx(tf, tr);
                const int o = owner(occupied, occupied_white, to_sq);
                if (o == SQ_EMPTY) {
                    moves.push_back(build_quiet(from_sq, to_sq, top));
                    continue;
                }
                if (o == SQ_BLACK && stacks.count((uint8_t)to_sq)) {  // friendly merge: emit + stop, capped
                    const auto& existing = stacks.at((uint8_t)to_sq);
                    if ((int)existing.size() + height <= MAX_TOWER_HEIGHT)
                        moves.push_back(build_quiet(from_sq, to_sq, top));
                    break;
                }
                break;  // White piece (or any non-friendly): blocks the slide; no quiet landing past it.
            }
        }
        // Sprint: height-1 unmoved Stone-top on rank 8, two squares forward.
        if (height == 1 && top == 's' && from_rank == 7) {
            for (auto [df, dr] : {std::pair<int, int>{-1, -1}, std::pair<int, int>{1, -1}}) {
                const int int_f = from_file + df, int_r = from_rank + dr;
                if (!on_board(int_f, int_r)) continue;
                if (owner(occupied, occupied_white, sq_idx(int_f, int_r)) != SQ_EMPTY) continue;
                const int tf = from_file + 2 * df, tr = from_rank + 2 * dr;
                if (!on_board(tf, tr)) continue;
                const int to_sq = sq_idx(tf, tr);
                const int o = owner(occupied, occupied_white, to_sq);
                if (o == SQ_EMPTY || (o == SQ_BLACK && stacks.count((uint8_t)to_sq) &&
                                       (int)stacks.at((uint8_t)to_sq).size() + 1 <= MAX_TOWER_HEIGHT))
                    moves.push_back(build_quiet(from_sq, to_sq, top));
            }
        }
    }
    return moves;
}

// -------- Deploys (Slice 2b) --------

struct DeployMove {
    std::string uci;
    uint8_t from_sq = 0, to_sq = 0;
    bool is_king = false;  // "king" vs "pawn"
    int deploy_count = 0;
};

inline DeployMove build_deploy(int from_sq, int to_sq, char top, int s) {
    DeployMove m;
    m.uci = square_name(from_sq) + square_name(to_sq) + "[" + std::to_string(s) + "]";
    m.from_sq = (uint8_t)from_sq;
    m.to_sq = (uint8_t)to_sq;
    m.is_king = (top == 'k');
    m.deploy_count = s;
    return m;
}

inline std::vector<DeployMove> black_deploy_moves(uint64_t occupied, uint64_t occupied_white,
                                                  const StackMap& stacks) {
    std::vector<DeployMove> moves;
    for (const auto& [from_sq, pieces] : stacks) {
        const int n = (int)pieces.size();
        if (n < 2) continue;
        const char top = pieces.back();
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        for (int s = 1; s < n; ++s) {
            for (auto [df, dr] : dirs_for_top(top)) {
                for (int k = 1; k <= s; ++k) {
                    const int tf = from_file + k * df, tr = from_rank + k * dr;
                    if (!on_board(tf, tr)) break;
                    const int to_sq = sq_idx(tf, tr);
                    const int o = owner(occupied, occupied_white, to_sq);
                    if (o == SQ_EMPTY) {
                        moves.push_back(build_deploy(from_sq, to_sq, top, s));
                        continue;
                    }
                    if (o == SQ_BLACK && stacks.count((uint8_t)to_sq)) {
                        const auto& existing = stacks.at((uint8_t)to_sq);
                        if ((int)existing.size() + s <= MAX_TOWER_HEIGHT)
                            moves.push_back(build_deploy(from_sq, to_sq, top, s));
                    }
                    break;
                }
            }
        }
    }
    return moves;
}

// -------- Charges (Slice 2c) --------

struct ChargeMove {
    std::string uci;
    uint8_t from_sq = 0, to_sq = 0;
    bool is_king = false;                                    // "king" vs "pawn"
    int16_t capture_sq = -1;                                 // 8x8 square; -1 == none
    bool has_waypoints = false;                              // null-vs-present (JSON parity)
    std::vector<uint8_t> waypoints;                          // [rim c10] for overshoot charge
    std::optional<std::vector<int>> demoted_kings;           // chosen king positions (1-based)
    std::optional<int> demotions_required;
    std::optional<std::vector<int>> source_king_positions;
};

inline std::vector<ChargeMove> black_charge_moves(uint64_t occupied, uint64_t occupied_white,
                                                  const StackMap& stacks) {
    std::vector<ChargeMove> moves;
    for (const auto& [from_sq, pieces] : stacks) {
        if (pieces.empty() || pieces.back() != 'k') continue;  // King-top towers only
        int n_kings = 0;
        for (char c : pieces)
            if (c == 'k') ++n_kings;
        if (n_kings == 0) continue;
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        std::vector<int> king_positions;  // 1-based indices of kings in the tower
        for (int i = 0; i < (int)pieces.size(); ++i)
            if (pieces[i] == 'k') king_positions.push_back(i + 1);

        for (auto [df, dr] : ORTHO_DIRS) {
            bool stop_after = false;
            for (int d = 1; d <= n_kings; ++d) {
                if (stop_after) break;
                // Path scan over intermediate squares 1..d-1.
                bool blocked = false, off_grid = false;
                std::vector<int> path_captures;
                int last_on_board_sq = -1;
                for (int k = 1; k < d; ++k) {
                    const int pf = from_file + k * df, pr = from_rank + k * dr;
                    if (pf < -1 || pf > 8 || pr < -1 || pr > 8) {
                        off_grid = true;
                        break;
                    }
                    if (pf >= 0 && pf <= 7 && pr >= 0 && pr <= 7) {
                        const int psq = sq_idx(pf, pr);
                        const int po = owner(occupied, occupied_white, psq);
                        if (po == SQ_BLACK && stacks.count((uint8_t)psq)) {
                            blocked = true;
                            break;
                        }
                        if (po == SQ_WHITE) path_captures.push_back(psq);
                        last_on_board_sq = psq;
                    }
                    // else: rim square, no action
                }
                if (off_grid) break;
                if (blocked) break;

                const int tf = from_file + d * df, tr = from_rank + d * dr;
                if (tf < -1 || tf > 8 || tr < -1 || tr > 8) break;  // off-grid landing

                bool is_ram = false, is_friendly_merge = false;
                int landing_sq = -1;
                int to_sq;  // the settled 8x8 square ("to" in dict/uci terms)
                bool has_rim_landing = false;
                uint8_t rim_landing_c10 = 0;
                if (tf >= 0 && tf <= 7 && tr >= 0 && tr <= 7) {
                    landing_sq = sq_idx(tf, tr);
                    to_sq = landing_sq;
                    const int o = owner(occupied, occupied_white, landing_sq);
                    is_ram = (o == SQ_WHITE);
                    is_friendly_merge = (o == SQ_BLACK && stacks.count((uint8_t)landing_sq));
                } else {
                    // Rim landing -> fall back to the last on-board square.
                    if (last_on_board_sq < 0) continue;  // d=1 rim: nothing to settle on
                    to_sq = last_on_board_sq;
                    has_rim_landing = true;
                    rim_landing_c10 = coord10_of(tf, tr);
                }

                int capture_sq = -1;
                if (!path_captures.empty()) capture_sq = path_captures[0];
                else if (is_ram) capture_sq = to_sq;

                if (is_ram) {
                    // §3C: a ram requires >=1 path capture (must overshoot an enemy).
                    if (!path_captures.empty()) {
                        ChargeMove m;
                        m.uci = square_name(from_sq) + square_name(to_sq);
                        m.from_sq = (uint8_t)from_sq;
                        m.to_sq = (uint8_t)to_sq;
                        m.is_king = true;
                        m.capture_sq = (int16_t)capture_sq;
                        moves.push_back(std::move(m));
                    }
                    continue;
                }

                // Friendly merge: only emit if combined height fits within cap.
                if (is_friendly_merge) {
                    const auto& existing = stacks.at((uint8_t)landing_sq);
                    if ((int)existing.size() + (int)pieces.size() > MAX_TOWER_HEIGHT) continue;
                }

                // v6 rule change: a charge of distance d demotes the BOTTOM d
                // Kings (king_positions[:d]; king_positions is ascending from the
                // bottom). One move per (from->landing), no {choice} suffix —
                // removes the old C(n_kings,d) demotion fan-out while preserving
                // the upper Kings. When n_kings==d this is the old demote-all case.
                // Mirrors PyVariant black_charge_moves exactly (parity).
                {
                    std::vector<int> chosen(king_positions.begin(),
                                            king_positions.begin() + d);
                    Tower new_pieces = pieces;
                    for (int pos : chosen) new_pieces[pos - 1] = 'S';
                    ChargeMove m;
                    // rim overshoot uci is `<from><rimkey>-><to>`, e.g. e2e0->e1
                    m.uci = square_name(from_sq);
                    if (has_rim_landing) {
                        m.uci += key_str(rim_landing_c10);
                        m.uci += "->";
                    }
                    m.uci += square_name(to_sq);
                    m.from_sq = (uint8_t)from_sq;
                    m.to_sq = (uint8_t)to_sq;
                    m.is_king = (new_pieces.back() == 'k');
                    m.capture_sq = (int16_t)capture_sq;
                    if (has_rim_landing) {
                        m.has_waypoints = true;
                        m.waypoints = {rim_landing_c10};
                    }
                    m.demoted_kings = chosen;
                    m.demotions_required = d;
                    m.source_king_positions = king_positions;
                    moves.push_back(std::move(m));
                }

                if (is_friendly_merge) stop_after = true;
            }
        }
    }
    return moves;
}

// -------- Mandate + assembly (Slice 2d) --------

// §4 mandate trigger: some Black tower has a diagonal-adjacent White and a
// non-suicide hop from there that lands on a board square. Mirrors scalachess
// hasMandatoryCapture (and the Rust/Python ports) including the adjacency
// pre-filter. Bool early-exit, so stack iteration order is irrelevant.
inline bool black_mandatory_capture_active(uint64_t occupied, uint64_t occupied_white,
                                           const StackMap& stacks) {
    for (const auto& [from_sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        const int n = (int)pieces.size();
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        for (auto [df, dr] : dirs_for_top(pieces.back())) {
            const int adj_f = from_file + df, adj_r = from_rank + dr;
            if (!on_board(adj_f, adj_r)) continue;
            if (owner(occupied, occupied_white, sq_idx(adj_f, adj_r)) != SQ_WHITE) continue;
            for (auto& hop :
                 find_capture_hops(occupied, occupied_white, stacks, from_file, from_rank, df, dr, n))
                if (!hop.is_suicide && hop.landing_square >= 0) return true;
        }
    }
    return false;
}

using AnyMove = std::variant<QuietMove, DeployMove, ChargeMove, ChainMove>;

// Full Black legal move list with the mandate filter applied. Order matches the
// Rust all_black_legal_moves_native exactly (the authoritative move order the
// policy head indexes): under mandate, charges-with-capture then chains;
// otherwise quiets, deploys, charges, chains.
inline std::vector<AnyMove> all_black_legal_moves(uint64_t occupied, uint64_t occupied_white,
                                                  long king_sq,
                                                  const StackMap& stacks) {
    auto quiet = black_diagonal_quiet_moves(occupied, occupied_white, stacks);
    auto deploy = black_deploy_moves(occupied, occupied_white, stacks);
    auto charge = black_charge_moves(occupied, occupied_white, stacks);
    auto chain = black_diagonal_capture_moves(occupied, occupied_white, king_sq, stacks);
    const bool mandate = black_mandatory_capture_active(occupied, occupied_white, stacks);

    std::vector<AnyMove> out;
    if (mandate) {
        for (auto& c : charge)
            if (c.capture_sq >= 0) out.push_back(c);
        for (auto& cm : chain) out.push_back(cm);
    } else {
        for (auto& q : quiet) out.push_back(q);
        for (auto& dm : deploy) out.push_back(dm);
        for (auto& c : charge) out.push_back(c);
        for (auto& cm : chain) out.push_back(cm);
    }
    return out;
}

// -------- White-king check predicate (Slice 3a) --------
//
// Used by White move legality: white_in_chessckers_check = can Black capture the
// White king via a diagonal chain/ram (black_can_capture_white_king, full chain
// search) OR is the king's square attacked under the cheaper walk-based model
// (square_attacked_by_black_chessckers). python-chess's own is_check is wrong
// here because it treats the Black-King encoding as a FIDE 8-direction king.

inline bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// Bool early-exit mirror of enumerate_chains_recursive: does any chain from here
// path-capture the king?
inline bool chain_captures_king_rec(uint64_t occupied, uint64_t occupied_white,
                                    const StackMap& stacks, int cf, int cr,
                                    const Tower& cur_stack, bool has_last_dir, int ldf, int ldr,
                                    bool has_cadence, int cadence, int n, int king) {
    if ((occupied_white & (1ULL << king)) == 0) return false;
    for (auto& hop : next_capture_options(occupied, occupied_white, stacks, cf, cr, cur_stack,
                                          has_last_dir, ldf, ldr, n, has_cadence, cadence, false)) {
        if (contains(hop.captures, king)) return true;
        if (hop.is_overshoot) continue;
        auto ap = apply_hop(occupied, occupied_white, stacks, cf, cr, cur_stack, hop);
        int nf, nr;
        if (hop.landing_square >= 0) {
            nf = hop.landing_square & 7;
            nr = hop.landing_square >> 3;
        } else {
            nf = c10_file(hop.landing_c10);
            nr = c10_rank(hop.landing_c10);
        }
        const int next_cadence = has_cadence ? cadence : hop.cadence;
        if (chain_captures_king_rec(ap.occupied, ap.occupied_white, ap.stacks, nf, nr, ap.land_stack,
                                    true, hop.df, hop.dr, true, next_cadence, n, king))
            return true;
    }
    return false;
}

inline bool black_can_capture_white_king(uint64_t occupied, uint64_t occupied_white, long king_sq,
                                         const StackMap& stacks) {
    if (king_sq < 0 || (occupied_white & (1ULL << king_sq)) == 0) return false;
    const int king = (int)king_sq;
    for (const auto& [sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        const int n = (int)pieces.size();
        const int cf = sq & 7, cr = sq >> 3;
        if (chain_captures_king_rec(occupied, occupied_white, stacks, cf, cr, pieces, false, 0, 0,
                                    false, 0, n, king))
            return true;
        // First-hop rams capture their path Whites in transit (may include the king).
        for (auto [df, dr] : dirs_for_top(pieces.back()))
            for (auto& hop : find_capture_hops(occupied, occupied_white, stacks, cf, cr, df, dr, n))
                if (hop.is_suicide && contains(hop.captures, king)) return true;
    }
    return false;
}

// Cheaper walk-based attack test on a target square (does NOT model rim-bounce
// diagonals — matches the Python/Rust reference). Diagonal walks (range = tower
// height; Whites in path don't block, friendly Black towers do) + orthogonal
// charges (King-top, n_kings>=2; rim-overshoot still attacks).
inline bool square_attacked_by_black_chessckers(uint64_t occupied, uint64_t occupied_white,
                                                const StackMap& stacks,
                                                int target_sq) {
    for (const auto& [from_sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        const int n = (int)pieces.size();
        const char top = pieces.back();
        const bool is_king_top = (top == 'k');
        int n_kings = 0;
        if (is_king_top)
            for (char c : pieces)
                if (c == 'k') ++n_kings;
        const int sf = from_sq & 7, sr = from_sq >> 3;

        for (auto [df, dr] : dirs_for_top(top)) {  // ALL_DIAGS for king-top, else FORWARD_DIAGS
            for (int k = 1; k <= n; ++k) {
                const int nf = sf + k * df, nr = sr + k * dr;
                if (!on_board(nf, nr)) break;
                const int nsq = sq_idx(nf, nr);
                if (nsq == target_sq) return true;
                if (owner(occupied, occupied_white, nsq) == SQ_BLACK && stacks.count((uint8_t)nsq))
                    break;
            }
        }

        if (is_king_top && n_kings >= 2) {
            for (auto [df, dr] : ORTHO_DIRS) {
                for (int k = 1; k < n_kings; ++k) {
                    const int nf = sf + k * df, nr = sr + k * dr;
                    if (!on_board(nf, nr)) break;
                    const int nsq = sq_idx(nf, nr);
                    if (nsq == target_sq) {
                        // charge to k+1 must land on the grid (board OR rim) to capture in transit
                        if (on_grid(sf + (k + 1) * df, sr + (k + 1) * dr)) return true;
                        break;
                    }
                    if (owner(occupied, occupied_white, nsq) == SQ_BLACK && stacks.count((uint8_t)nsq))
                        break;
                }
            }
        }
    }
    return false;
}

inline bool white_in_chessckers_check(uint64_t occupied, uint64_t occupied_white, long white_king,
                                      const StackMap& stacks) {
    if (white_king < 0) return false;  // king already captured
    if (black_can_capture_white_king(occupied, occupied_white, white_king, stacks)) return true;
    return square_attacked_by_black_chessckers(occupied, occupied_white, stacks, (int)white_king);
}

}  // namespace cc
