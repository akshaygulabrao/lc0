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

// (file, rank) on the 10×10 grid -> 2-char key. Rim files use 'z'/'i' for
// -1/8, rim ranks use '0'/'9' for -1/8. Matches Python _COORD_KEY / Rust coord_key.
inline std::string coord_key(int f, int r) {
    char fc;
    if (f == -1) fc = 'z';
    else if (f >= 0 && f <= 7) fc = char('a' + f);
    else if (f == 8) fc = 'i';
    else fc = '?';
    std::string s;
    s += fc;
    s += char('0' + (r + 1));  // r in [-1,8] -> digit '0'..'9'
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
    std::string key;
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
                            steps.push_back(PathStep{f, r, sq, coord_key(f, r), df0, dr0, false});
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
    std::string landing_key;
    int landing_square;                  // -1 == None (rim / overshoot)
    std::vector<int> captures;           // board squares of Whites captured on the path
    std::vector<std::string> waypoints;  // every traced step's key (incl. landing)
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
    const std::map<uint8_t, std::string>& stacks,
    int f0, int r0, int df0, int dr0, int n) {
    std::vector<CaptureHop> options;
    std::vector<int> captures_so_far;
    uint64_t captured_set = 0;
    std::vector<std::string> waypoints_so_far;
    bool crossed_rank1 = false;
    bool friendly_blocked = false;

    const auto it = capture_paths().find({f0, r0, df0, dr0});
    if (it == capture_paths().end()) return options;
    const std::vector<PathStep>& path = it->second;
    const int max_step = n + 1;

    for (int step_idx = 0; step_idx < (int)path.size(); ++step_idx) {
        if (step_idx >= max_step) break;
        const PathStep& step = path[step_idx];
        const std::string& cur_key = step.key;
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

inline std::optional<std::pair<int, int>> parse_waypoint_key(const std::string& s) {
    if (s.size() != 2) return std::nullopt;
    const char fc = s[0], rc = s[1];
    int f;
    if (fc == 'z') f = -1;
    else if (fc >= 'a' && fc <= 'h') f = fc - 'a';
    else if (fc == 'i') f = 8;
    else return std::nullopt;
    if (rc < '0' || rc > '9') return std::nullopt;
    return std::make_pair(f, (rc - '0') - 1);
}

inline std::vector<std::pair<int, int>> dirs_for_top(char top) {
    if (top == 'k') return {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    return {{-1, -1}, {1, -1}};
}

inline bool hop_promotes(const CaptureHop& hop) {
    if (hop.crossed_rank1) return true;
    if (hop.landing_square >= 0 && (hop.landing_square >> 3) == 0) return true;
    return false;
}

inline std::string promote_all_stones(const std::string& stack) {
    std::string out = stack;
    for (char& c : out)
        if (c == 's' || c == 'S') c = 'k';
    return out;
}

inline std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

struct ChainMove {
    std::string uci, from_name, to_name;
    std::string piece;  // "king" / "pawn"
    std::optional<std::string> capture;
    std::optional<std::vector<std::string>> waypoints;
    std::vector<std::string> chain_hops;
    std::vector<std::string> chain_all_captures;
    bool is_suicide;
    bool chain_promotes;
    int cadence;
};

// dedup + cadence-lock + last-dir(no-reversal) + suicide filter over the hop
// atom, across the directions valid for the current tower top.
inline std::vector<CaptureHop> next_capture_options(
    uint64_t occupied, uint64_t occupied_white, const std::map<uint8_t, std::string>& stacks,
    int cf, int cr, const std::string& cur_stack, bool has_last_dir, int ldf, int ldr, int n,
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
    // identity = (df, dr, landing_key, captures, is_suicide, is_overshoot, cadence)
    std::set<std::tuple<int, int, std::string, std::vector<int>, bool, bool, int>> seen;
    std::vector<CaptureHop> deduped;
    for (auto& h : options) {
        auto key = std::make_tuple(h.df, h.dr, h.landing_key, h.captures, h.is_suicide,
                                   h.is_overshoot, h.cadence);
        if (seen.insert(key).second) deduped.push_back(h);
    }
    return deduped;
}

inline ChainMove build_final_move(int chain_start, const std::string& orig_stack,
                                  const std::vector<CaptureHop>& hops) {
    const bool is_suicide_chain = !hops.empty() && hops.back().is_suicide;
    std::vector<int> all_captures;
    std::vector<std::string> all_waypoints, hop_keys;
    for (auto& h : hops) {
        all_captures.insert(all_captures.end(), h.captures.begin(), h.captures.end());
        all_waypoints.insert(all_waypoints.end(), h.waypoints.begin(), h.waypoints.end());
        hop_keys.push_back(h.landing_key);
    }

    const int last_landing = hops.empty() ? -1 : hops.back().landing_square;
    int final_landing;
    if (last_landing >= 0) {
        final_landing = last_landing;
    } else {
        // End-of-turn fallback: last on-board waypoint, else the chain start.
        final_landing = chain_start;
        for (auto it = all_waypoints.rbegin(); it != all_waypoints.rend(); ++it) {
            const auto pr = parse_waypoint_key(*it);
            if (pr && on_board(pr->first, pr->second)) {
                final_landing = sq_idx(pr->first, pr->second);
                break;
            }
        }
    }

    char final_top;
    if (is_suicide_chain) {
        final_top = orig_stack.back();
    } else {
        std::string stack_thru = orig_stack;
        for (auto& h : hops)
            if (hop_promotes(h)) stack_thru = promote_all_stones(stack_thru);
        final_top = stack_thru.back();
    }

    const std::string from_name = square_name(chain_start);
    const std::string dest_name = square_name(final_landing);

    std::optional<std::string> capture;
    if (!all_captures.empty()) capture = square_name(all_captures[0]);
    else if (is_suicide_chain) capture = square_name(final_landing);

    const int cadence = hops[0].cadence;
    const std::string uci =
        "c" + std::to_string(cadence) + ":" + from_name + "~" + join(hop_keys, "~") + "->" + dest_name;

    std::vector<std::string> all_cap_names;
    for (int sq : all_captures) all_cap_names.push_back(square_name(sq));
    bool chain_promotes_any = false;
    for (auto& h : hops)
        if (hop_promotes(h)) {
            chain_promotes_any = true;
            break;
        }

    ChainMove m;
    m.uci = uci;
    m.from_name = from_name;
    m.to_name = dest_name;
    m.piece = (final_top == 'k') ? "king" : "pawn";
    m.capture = capture;
    if (hops.size() > 1) m.waypoints = all_waypoints;
    m.chain_hops = hop_keys;
    m.chain_all_captures = all_cap_names;
    m.is_suicide = is_suicide_chain;
    m.chain_promotes = chain_promotes_any;
    m.cadence = cadence;
    return m;
}

struct HopApply {
    uint64_t occupied, occupied_white;
    std::map<uint8_t, std::string> stacks;
    std::string land_stack;
};

inline HopApply apply_hop(uint64_t occupied, uint64_t occupied_white,
                          std::map<uint8_t, std::string> stacks, int cf, int cr,
                          const std::string& cur_stack, const CaptureHop& hop) {
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
    std::string land_stack = hop_promotes(hop) ? promote_all_stones(cur_stack) : cur_stack;
    if (hop.landing_square >= 0) {
        const uint64_t mask = 1ULL << hop.landing_square;
        occupied |= mask;
        occupied_white &= ~mask;  // Black
        stacks[(uint8_t)hop.landing_square] = land_stack;
    }
    return {occupied, occupied_white, std::move(stacks), std::move(land_stack)};
}

inline void enumerate_chains_recursive(uint64_t occupied, uint64_t occupied_white, long king_sq,
                                       const std::map<uint8_t, std::string>& stacks, int chain_start,
                                       int cf, int cr, const std::string& cur_stack,
                                       bool has_last_dir, int ldf, int ldr,
                                       std::vector<CaptureHop> hops_so_far, bool has_cadence,
                                       int cadence, int n, const std::string& orig_stack,
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
            const auto pr = parse_waypoint_key(hop.landing_key);
            nf = pr ? pr->first : cf;
            nr = pr ? pr->second : cr;
        }
        const int next_cadence = has_cadence ? cadence : hop.cadence;
        enumerate_chains_recursive(ap.occupied, ap.occupied_white, king_sq, ap.stacks, chain_start,
                                   nf, nr, ap.land_stack, true, hop.df, hop.dr, hops_next, true,
                                   next_cadence, n, orig_stack, results);
    }
}

inline std::vector<ChainMove> enumerate_chains(uint64_t occupied, uint64_t occupied_white,
                                               long king_sq,
                                               const std::map<uint8_t, std::string>& stacks,
                                               int chain_start) {
    const auto it = stacks.find((uint8_t)chain_start);
    if (it == stacks.end() || it->second.empty()) return {};
    const std::string orig_stack = it->second;
    const int n = (int)orig_stack.size();
    std::vector<ChainMove> results;
    enumerate_chains_recursive(occupied, occupied_white, king_sq, stacks, chain_start,
                               chain_start & 7, chain_start >> 3, orig_stack, false, 0, 0, {}, false,
                               0, n, orig_stack, results);
    return results;
}

inline std::vector<ChainMove> first_hop_suicides(uint64_t occupied, uint64_t occupied_white,
                                                 const std::map<uint8_t, std::string>& stacks,
                                                 int chain_start) {
    const auto it = stacks.find((uint8_t)chain_start);
    if (it == stacks.end() || it->second.empty()) return {};
    const std::string pieces = it->second;
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
                                                           const std::map<uint8_t, std::string>& stacks) {
    std::vector<ChainMove> moves;
    // std::map iterates ascending key order -> matches Rust keys.sort_unstable().
    for (auto& [sq, pieces] : stacks) {
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
    std::string uci, from_name, to_name, piece;
};

inline QuietMove build_quiet(const std::string& from_name, int to_sq, char top) {
    const std::string to_name = square_name(to_sq);
    return {from_name + to_name, from_name, to_name, (top == 'k') ? "king" : "pawn"};
}

inline std::vector<QuietMove> black_diagonal_quiet_moves(uint64_t occupied, uint64_t occupied_white,
                                                         const std::map<uint8_t, std::string>& stacks) {
    std::vector<QuietMove> moves;
    for (auto& [from_sq, pieces] : stacks) {
        if (pieces.empty()) continue;
        const int height = (int)pieces.size();
        const char top = pieces.back();
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        const std::string from_name = square_name(from_sq);

        for (auto [df, dr] : dirs_for_top(top)) {
            for (int k = 1; k <= height; ++k) {
                const int tf = from_file + k * df, tr = from_rank + k * dr;
                if (!on_board(tf, tr)) break;
                const int to_sq = sq_idx(tf, tr);
                const int o = owner(occupied, occupied_white, to_sq);
                if (o == SQ_EMPTY) {
                    moves.push_back(build_quiet(from_name, to_sq, top));
                    continue;
                }
                if (o == SQ_BLACK && stacks.count((uint8_t)to_sq))  // friendly merge: emit + stop
                    moves.push_back(build_quiet(from_name, to_sq, top));
                break;
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
                if (o == SQ_EMPTY || (o == SQ_BLACK && stacks.count((uint8_t)to_sq)))
                    moves.push_back(build_quiet(from_name, to_sq, top));
            }
        }
    }
    return moves;
}

// -------- Deploys (Slice 2b) --------

struct DeployMove {
    std::string uci, from_name, to_name, piece;
    int deploy_count;
};

inline DeployMove build_deploy(const std::string& from_name, int to_sq, char top, int s) {
    const std::string to_name = square_name(to_sq);
    return {from_name + to_name + "[" + std::to_string(s) + "]", from_name, to_name,
            (top == 'k') ? "king" : "pawn", s};
}

inline std::vector<DeployMove> black_deploy_moves(uint64_t occupied, uint64_t occupied_white,
                                                  const std::map<uint8_t, std::string>& stacks) {
    std::vector<DeployMove> moves;
    for (auto& [from_sq, pieces] : stacks) {
        const int n = (int)pieces.size();
        if (n < 2) continue;
        const char top = pieces.back();
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        const std::string from_name = square_name(from_sq);
        for (int s = 1; s < n; ++s) {
            for (auto [df, dr] : dirs_for_top(top)) {
                for (int k = 1; k <= s; ++k) {
                    const int tf = from_file + k * df, tr = from_rank + k * dr;
                    if (!on_board(tf, tr)) break;
                    const int to_sq = sq_idx(tf, tr);
                    const int o = owner(occupied, occupied_white, to_sq);
                    if (o == SQ_EMPTY) {
                        moves.push_back(build_deploy(from_name, to_sq, top, s));
                        continue;
                    }
                    if (o == SQ_BLACK && stacks.count((uint8_t)to_sq))
                        moves.push_back(build_deploy(from_name, to_sq, top, s));
                    break;
                }
            }
        }
    }
    return moves;
}

// -------- Charges (Slice 2c) --------

struct ChargeMove {
    std::string uci, from_name, to_name, piece;
    std::optional<std::string> capture;
    std::optional<std::vector<std::string>> waypoints;       // [rim key] for overshoot charge
    std::optional<std::vector<int>> demoted_kings;           // chosen king positions (1-based)
    std::optional<int> demotions_required;
    std::optional<std::vector<int>> source_king_positions;
};

// In-place r-combinations of `items` (lexicographic). 1:1 with the Rust port —
// used to enumerate king-demotion choices when n_kings > d.
inline std::vector<std::vector<int>> combinations(const std::vector<int>& items, int r) {
    const int n = (int)items.size();
    if (r == 0 || r > n) return {};
    std::vector<std::vector<int>> out;
    std::vector<int> idx(r);
    for (int i = 0; i < r; ++i) idx[i] = i;
    while (true) {
        std::vector<int> combo;
        combo.reserve(r);
        for (int i = 0; i < r; ++i) combo.push_back(items[idx[i]]);
        out.push_back(std::move(combo));
        int i = r;
        while (i > 0) {
            --i;
            if (idx[i] != i + n - r) break;
            if (i == 0) return out;
        }
        idx[i] += 1;
        for (int j = i + 1; j < r; ++j) idx[j] = idx[j - 1] + 1;
        if (idx[0] > n - r) break;
    }
    return out;
}

inline std::vector<ChargeMove> black_charge_moves(uint64_t occupied, uint64_t occupied_white,
                                                  const std::map<uint8_t, std::string>& stacks) {
    std::vector<ChargeMove> moves;
    for (auto& [from_sq, pieces] : stacks) {
        if (pieces.empty() || pieces.back() != 'k') continue;  // King-top towers only
        int n_kings = 0;
        for (char c : pieces)
            if (c == 'k') ++n_kings;
        if (n_kings == 0) continue;
        const int from_file = from_sq & 7, from_rank = from_sq >> 3;
        const std::string from_name = square_name(from_sq);
        std::vector<int> king_positions;  // 1-based indices of kings in the tower
        for (int i = 0; i < (int)pieces.size(); ++i)
            if (pieces[i] == 'k') king_positions.push_back(i + 1);

        for (auto [df, dr] : ORTHO_DIRS) {
            bool stop_after = false;
            for (int d = 1; d <= n_kings; ++d) {
                if (stop_after) break;
                // Path scan over intermediate squares 1..d-1.
                bool blocked = false, off_grid = false;
                std::vector<std::string> path_captures;
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
                        if (po == SQ_WHITE) path_captures.push_back(square_name(psq));
                        last_on_board_sq = psq;
                    }
                    // else: rim square, no action
                }
                if (off_grid) break;
                if (blocked) break;

                const int tf = from_file + d * df, tr = from_rank + d * dr;
                if (tf < -1 || tf > 8 || tr < -1 || tr > 8) break;  // off-grid landing

                std::string to_name;
                bool is_ram = false, is_friendly_merge = false;
                std::optional<std::string> rim_landing_key;
                if (tf >= 0 && tf <= 7 && tr >= 0 && tr <= 7) {
                    const int s = sq_idx(tf, tr);
                    to_name = square_name(s);
                    const int o = owner(occupied, occupied_white, s);
                    is_ram = (o == SQ_WHITE);
                    is_friendly_merge = (o == SQ_BLACK && stacks.count((uint8_t)s));
                } else {
                    // Rim landing -> fall back to the last on-board square.
                    if (last_on_board_sq < 0) continue;  // d=1 rim: nothing to settle on
                    to_name = square_name(last_on_board_sq);
                    rim_landing_key = coord_key(tf, tr);
                }

                std::string landing_repr;
                std::optional<std::vector<std::string>> charge_waypoints;
                if (!rim_landing_key) {
                    landing_repr = to_name;
                } else {
                    landing_repr = *rim_landing_key + "->" + to_name;  // e.g. e0->e1
                    charge_waypoints = std::vector<std::string>{*rim_landing_key};
                }

                std::optional<std::string> capture_field;
                if (!path_captures.empty()) capture_field = path_captures[0];
                else if (is_ram) capture_field = to_name;

                if (is_ram) {
                    // §3C: a ram requires >=1 path capture (must overshoot an enemy).
                    if (!path_captures.empty()) {
                        ChargeMove m;
                        m.uci = from_name + to_name;
                        m.from_name = from_name;
                        m.to_name = to_name;
                        m.piece = "king";
                        m.capture = capture_field;
                        moves.push_back(std::move(m));
                    }
                    continue;
                }

                if (n_kings == d) {
                    // Forced demotion (all kings) -> null choice fields.
                    std::string new_pieces = pieces;
                    for (int pos : king_positions) new_pieces[pos - 1] = 'S';
                    ChargeMove m;
                    m.uci = from_name + landing_repr;
                    m.from_name = from_name;
                    m.to_name = to_name;
                    m.piece = (new_pieces.back() == 'k') ? "king" : "pawn";
                    m.capture = capture_field;
                    m.waypoints = charge_waypoints;
                    moves.push_back(std::move(m));
                } else {
                    for (auto& choice : combinations(king_positions, d)) {
                        std::string new_pieces = pieces;
                        for (int pos : choice) new_pieces[pos - 1] = 'S';
                        std::vector<std::string> cs;
                        for (int i : choice) cs.push_back(std::to_string(i));
                        ChargeMove m;
                        m.uci = from_name + landing_repr + "{" + join(cs, ",") + "}";
                        m.from_name = from_name;
                        m.to_name = to_name;
                        m.piece = (new_pieces.back() == 'k') ? "king" : "pawn";
                        m.capture = capture_field;
                        m.waypoints = charge_waypoints;
                        m.demoted_kings = choice;
                        m.demotions_required = d;
                        m.source_king_positions = king_positions;
                        moves.push_back(std::move(m));
                    }
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
                                           const std::map<uint8_t, std::string>& stacks) {
    for (auto& [from_sq, pieces] : stacks) {
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
                                                  const std::map<uint8_t, std::string>& stacks) {
    auto quiet = black_diagonal_quiet_moves(occupied, occupied_white, stacks);
    auto deploy = black_deploy_moves(occupied, occupied_white, stacks);
    auto charge = black_charge_moves(occupied, occupied_white, stacks);
    auto chain = black_diagonal_capture_moves(occupied, occupied_white, king_sq, stacks);
    const bool mandate = black_mandatory_capture_active(occupied, occupied_white, stacks);

    std::vector<AnyMove> out;
    if (mandate) {
        for (auto& c : charge)
            if (c.capture.has_value()) out.push_back(c);
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
                                    const std::map<uint8_t, std::string>& stacks, int cf, int cr,
                                    const std::string& cur_stack, bool has_last_dir, int ldf, int ldr,
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
            const auto pr = parse_waypoint_key(hop.landing_key);
            nf = pr ? pr->first : cf;
            nr = pr ? pr->second : cr;
        }
        const int next_cadence = has_cadence ? cadence : hop.cadence;
        if (chain_captures_king_rec(ap.occupied, ap.occupied_white, ap.stacks, nf, nr, ap.land_stack,
                                    true, hop.df, hop.dr, true, next_cadence, n, king))
            return true;
    }
    return false;
}

inline bool black_can_capture_white_king(uint64_t occupied, uint64_t occupied_white, long king_sq,
                                         const std::map<uint8_t, std::string>& stacks) {
    if (king_sq < 0 || (occupied_white & (1ULL << king_sq)) == 0) return false;
    const int king = (int)king_sq;
    for (auto& [sq, pieces] : stacks) {
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
                                                const std::map<uint8_t, std::string>& stacks,
                                                int target_sq) {
    for (auto& [from_sq, pieces] : stacks) {
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
                                      const std::map<uint8_t, std::string>& stacks) {
    if (white_king < 0) return false;  // king already captured
    if (black_can_capture_white_king(occupied, occupied_white, white_king, stacks)) return true;
    return square_attacked_by_black_chessckers(occupied, occupied_white, stacks, (int)white_king);
}

}  // namespace cc
