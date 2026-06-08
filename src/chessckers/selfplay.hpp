// Chessckers C++ engine — Phase 2 (lc0-split migration): pure-C++ self-play.
//
// The GIL-free counterpart of bindings.cpp's play_game_native. Every step —
// legal-move gen, leaf expansion, NN forward, apply, sample, record — runs on
// NativeMoves with ZERO pybind in the hot loop, so play_games_pure can fan games
// across std::threads under py::gil_scoped_release. The tree math is the same
// search.hpp; the only change from the Phase-1 path is the move representation
// (gen_legal_native / encode_native_move / apply_native instead of the dict
// round-trip). A deterministic pure game is byte-identical to play_game_native —
// that equivalence is the Phase-2 parity test.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "apply.hpp"
#include "native_move.hpp"
#include "search.hpp"

namespace cc {

// One recorded ply: the position, its legal NativeMoves, the aligned visit
// counts, and the side to move. The Python layer turns this into an AZRecord
// (converting `legal` to dicts under the GIL, after the threads join).
struct PureRecord {
    std::string fen;
    std::vector<NativeMove> legal;
    std::vector<int> visits;  // aligned with `legal`
    bool side_white;
};

struct PureGame {
    std::vector<PureRecord> records;
    std::string outcome;        // "white" / "black" / "draw"
    std::string final_status;   // "" == None (resigned/maxplies games)
};

// ---- leaf evaluator indirection (Phase 6d: GPU leaf batching) ----
// The NN FORWARD is the only thing that varies between the serial path and the batched
// GPU path; encoding stays inline (it needs the net's arch). DirectEvaluator is the
// byte-identical default (calls net.eval). BatchEvaluator gathers leaves from ALL
// in-flight games into ONE eval_batch — the lc0 design (many concurrent games, one
// shared batched backend). Because eval_batch is byte-identical to K serial eval()
// calls, every game played through the batcher is byte-identical to its serial self,
// no matter how the threads pack leaves into batches — the parity gate is NOT loosened.
struct Evaluator {
    virtual ~Evaluator() = default;
    virtual std::pair<float, std::vector<float>> eval(
        const std::vector<float>& pos, const std::vector<std::vector<float>>& moves) = 0;
};

struct DirectEvaluator : Evaluator {
    const ChesskersNet& net;
    explicit DirectEvaluator(const ChesskersNet& n) : net(n) {}
    std::pair<float, std::vector<float>> eval(
        const std::vector<float>& pos, const std::vector<std::vector<float>>& moves) override {
        return net.eval(pos, moves);
    }
};

// The batched TRUNK forward: K positions -> K feature maps [c_filters*100]. Either
// net.trunk_v2_batch (CPU BLAS) or MetalTrunkV2::run (GPU) — selfplay.hpp stays
// backend-agnostic (the binding supplies it; cc_selfplay supplies the CPU one), so Metal
// types never leak in here. Only the trunk batches; the per-board value/policy heads run
// on the GAME threads (Phase 6e) — they don't batch well and parallelize off the gatherer.
using TrunkForwardFn =
    std::function<std::vector<std::vector<float>>(const std::vector<std::vector<float>>&)>;

// Shared batching evaluator. Each game thread calls eval() and BLOCKS while the gatherer
// batches the trunk; on wake the thread computes its OWN head_v2 (value + policy) from its
// features, in parallel with the other game threads. The gatherer fires once every running
// game has submitted (or the cap is reached). V1 nets skip the gatherer entirely (their
// trunk doesn't have the batched path) and just eval serially on the calling thread.
struct BatchEvaluator : Evaluator {
    const ChesskersNet& net;
    TrunkForwardFn trunk_forward;
    int cap;                    // max leaves per batch (== concurrent games)
    std::mutex m;
    std::condition_variable cv_gather;
    int active;                 // games still able to submit (workers not yet exhausted)

    struct Req {
        const std::vector<float>* pos;
        std::vector<float> feat;  // trunk features, filled by the gatherer
        bool ready = false;
        std::condition_variable cv;
    };
    std::vector<Req*> pending;

    BatchEvaluator(const ChesskersNet& n, TrunkForwardFn f, int cap_, int active_)
        : net(n), trunk_forward(std::move(f)), cap(cap_), active(active_) {}

    std::pair<float, std::vector<float>> eval(
        const std::vector<float>& pos, const std::vector<std::vector<float>>& moves) override {
        if (!net.is_v2) return net.eval(pos, moves);  // V1: no batched trunk -> serial
        std::vector<float> feat;
        {
            std::unique_lock<std::mutex> lk(m);
            Req req;
            req.pos = &pos;
            pending.push_back(&req);
            cv_gather.notify_one();
            req.cv.wait(lk, [&] { return req.ready; });
            feat = std::move(req.feat);
        }
        return net.head_v2(feat, moves);  // head on THIS game thread (parallel), not the gatherer
    }

    // A worker calls this exactly once when it will never submit another leaf (its
    // game queue is exhausted) so the gatherer stops waiting on it.
    void worker_exit() {
        std::unique_lock<std::mutex> lk(m);
        --active;
        cv_gather.notify_one();
    }

    // Runs on the launching thread until every worker has exited. Funnels ALL forward
    // calls through this one thread (so the GPU command queue is single-threaded).
    void run_gatherer() {
        std::unique_lock<std::mutex> lk(m);
        while (true) {
            cv_gather.wait(lk, [&] {
                return active == 0 || (int)pending.size() >= active || (int)pending.size() >= cap;
            });
            if (pending.empty()) {
                if (active == 0) return;
                continue;
            }
            const int take = std::min((int)pending.size(), cap);
            std::vector<Req*> batch(pending.begin(), pending.begin() + take);
            pending.erase(pending.begin(), pending.begin() + take);
            std::vector<std::vector<float>> P;
            P.reserve(take);
            for (auto* r : batch) P.push_back(*r->pos);
            lk.unlock();
            auto feats = trunk_forward(P);  // the one batched trunk forward — GIL already released
            lk.lock();
            for (int i = 0; i < take; ++i) {
                batch[i]->feat = std::move(feats[i]);
                batch[i]->ready = true;
                batch[i]->cv.notify_one();
            }
        }
    }
};

// ---- leaf expansion (pure mirror of bindings.cpp mcts_expand_native) ----
inline double mcts_expand_pure(PuctNode* node, const ChesskersNet& net, Evaluator& ev) {
    auto legal = gen_legal_native(node->board);
    const size_t n = legal.size();
    node->expanded = true;
    if (n == 0) {  // no-moves terminal — resolve its exact status lazily
        const auto st = detect_status(node->board);
        node->is_terminal = true;
        node->terminal_status = st.status;
        return terminal_value(*node);
    }
    const auto pos_enc = net.is_v2 ? encode_position_v2(node->board) : encode_position(node->board);
    std::vector<std::vector<float>> move_encs;
    move_encs.reserve(n);
    for (const auto& m : legal) move_encs.push_back(encode_native_move(net, m));
    const auto r = ev.eval(pos_enc, move_encs);
    for (size_t i = 0; i < n; ++i) {
        auto child = std::make_unique<PuctNode>();
        child->board = apply_native(node->board, legal[i]);
        child->uci = legal[i].uci;
        child->prior = r.second[i];
        // Cheap-only terminal checks (popcounts) — the dominant Chessckers
        // terminals; no-moves terminals are caught lazily when expanded (n==0).
        if (child->board.stacks.empty() ||
            (child->board.kings & child->board.occupied_white) == 0) {
            child->is_terminal = true;
            child->terminal_status = "variantEnd";
        }
        node->children.push_back(std::move(child));
    }
    return r.first;
}

inline void mcts_simulate_pure(PuctNode* root, const ChesskersNet& net, Evaluator& ev,
                               double c_puct, double gamma) {
    const auto path = select_to_leaf(root, c_puct);
    PuctNode* leaf = path.back();
    double value;
    if (leaf->is_terminal) value = terminal_value(*leaf);
    else if (!leaf->expanded) value = mcts_expand_pure(leaf, net, ev);
    else value = ev.eval(net.is_v2 ? encode_position_v2(leaf->board) : encode_position(leaf->board),
                         {})
                     .first;
    backup(path, value, gamma);
}

// Expand root + Dirichlet root noise + search to the n_sims budget (carried reuse
// visits count). Pure mirror of bindings.cpp expand_dirichlet_search.
inline void expand_dirichlet_search_pure(PuctNode* root, const ChesskersNet& net, Evaluator& ev,
                                         int n_sims, double c_puct, double gamma,
                                         double dirichlet_alpha, double dirichlet_eps,
                                         std::mt19937_64& rng) {
    if (n_sims > 0 && !(root->expanded && !root->children.empty()))
        mcts_simulate_pure(root, net, ev, c_puct, gamma);
    if (dirichlet_alpha > 0.0 && !root->children.empty()) {
        std::gamma_distribution<double> gd(dirichlet_alpha, 1.0);
        const int nch = (int)root->children.size();
        std::vector<double> noise(nch);
        double s = 0.0;
        for (int i = 0; i < nch; ++i) {
            noise[i] = gd(rng);
            s += noise[i];
        }
        if (s > 0.0) {
            int i = 0;
            for (auto& c : root->children) {
                c->prior = (1.0 - dirichlet_eps) * c->prior + dirichlet_eps * (noise[i] / s);
                ++i;
            }
        }
    }
    const int remaining = std::max(0, n_sims - root->visits);
    for (int i = 0; i < remaining; ++i) mcts_simulate_pure(root, net, ev, c_puct, gamma);
}

// temp<=0 -> argmax (first max); temp>0 -> sample ∝ visits^(1/temp). Pure mirror
// of bindings.cpp sample_move_index.
inline int sample_move_index_pure(const std::vector<int>& visits, double temperature,
                                  std::mt19937_64& rng) {
    const int n = (int)visits.size();
    if (n == 0) return 0;
    if (temperature <= 0.0) {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (visits[i] > visits[best]) best = i;
        return best;
    }
    long total = 0;
    for (int v : visits) total += v;
    if (total == 0) return 0;
    std::vector<double> probs(n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::pow((double)visits[i], 1.0 / temperature);
        s += probs[i];
    }
    std::uniform_real_distribution<double> u(0.0, s);
    const double r = u(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return n - 1;
}

inline std::string outcome_from_pure(const std::string& status, const std::string& winner) {
    if (winner == "white") return "white";
    if (winner == "black") return "black";
    if (status == "mate") return "black";
    if (status == "variantEnd") return "white";
    return "draw";
}

// One fully-native self-play game. Byte-identical to play_game_native (same
// per-ply search, sample, apply, record, Lc0 tree reuse, resignation) but with no
// py:: anywhere — safe to call with the GIL released.
inline PureGame play_game_pure(Board board, const ChesskersNet& net, Evaluator& ev, int n_sims,
                               double c_puct, double temperature, int temp_cutoff_plies,
                               int max_plies, double dirichlet_alpha, double dirichlet_eps,
                               uint64_t seed, double resign_threshold, double resign_no_resign_frac,
                               int resign_consecutive, int resign_min_ply, double gamma) {
    std::mt19937_64 rng(seed);
    const bool resign_enabled = resign_threshold > 0.0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const bool no_resign_game = !resign_enabled || (u01(rng) < resign_no_resign_frac);
    int consec_resign = 0;

    PureGame game;
    std::unique_ptr<PuctNode> reuse;  // detached subtree from the previous ply
    int ply = 0;
    std::string final_status, final_winner;
    bool resigned = false;

    while (ply < max_plies) {
        const auto st = detect_status(board);
        if (!st.status.empty()) {
            final_status = st.status;
            final_winner = st.winner;
            break;
        }
        auto legal = gen_legal_native(board);
        const int nlegal = (int)legal.size();
        if (nlegal == 0) break;  // no moves but not flagged terminal — end as draw

        std::unique_ptr<PuctNode> root;
        if (reuse && serialize_fen(reuse->board) == serialize_fen(board)) {
            root = std::move(reuse);
            root->uci.clear();
        } else {
            root = std::make_unique<PuctNode>();
            root->board = board;
        }
        expand_dirichlet_search_pure(root.get(), net, ev, n_sims, c_puct, gamma, dirichlet_alpha,
                                     dirichlet_eps, rng);

        // Visit counts aligned to `legal` by uci (robust to any gen-order detail).
        std::map<std::string, int> vmap;
        for (auto& c : root->children) vmap[c->uci] = c->visits;
        std::vector<int> visits(nlegal, 0);
        for (int i = 0; i < nlegal; ++i) {
            const auto it = vmap.find(legal[i].uci);
            visits[i] = (it != vmap.end()) ? it->second : 0;
        }
        const double root_value = root->q();

        PureRecord rec;
        rec.fen = serialize_fen(board);
        rec.legal = legal;  // copy (the chosen move + reuse detach below mutate root, not legal)
        rec.visits = visits;
        rec.side_white = board.turn_white;
        game.records.push_back(std::move(rec));

        if (resign_enabled && !no_resign_game && ply >= resign_min_ply) {
            if (root_value <= -resign_threshold) {
                if (++consec_resign >= resign_consecutive) {
                    game.outcome = board.turn_white ? "black" : "white";  // STM resigns
                    resigned = true;
                    break;
                }
            } else {
                consec_resign = 0;
            }
        }

        const double eff_temp = (ply < temp_cutoff_plies) ? temperature : 0.0;
        const int idx = sample_move_index_pure(visits, eff_temp, rng);
        const std::string chosen_uci = legal[idx].uci;

        board = apply_native(board, legal[idx]);

        reuse.reset();  // Lc0 tree reuse: detach the chosen child to re-root next ply
        for (auto& c : root->children) {
            if (c && c->uci == chosen_uci) {
                reuse = std::move(c);
                break;
            }
        }
        ++ply;
    }

    if (!resigned) game.outcome = outcome_from_pure(final_status, final_winner);
    game.final_status = final_status;  // "" stays "" (None)
    return game;
}

// Serial overload: the byte-identical default path (DirectEvaluator -> net.eval). Every
// existing caller (play_games_pure, play_game_chunk, play_game_native) routes here.
inline PureGame play_game_pure(Board board, const ChesskersNet& net, int n_sims, double c_puct,
                               double temperature, int temp_cutoff_plies, int max_plies,
                               double dirichlet_alpha, double dirichlet_eps, uint64_t seed,
                               double resign_threshold, double resign_no_resign_frac,
                               int resign_consecutive, int resign_min_ply, double gamma) {
    DirectEvaluator ev(net);
    return play_game_pure(std::move(board), net, ev, n_sims, c_puct, temperature, temp_cutoff_plies,
                          max_plies, dirichlet_alpha, dirichlet_eps, seed, resign_threshold,
                          resign_no_resign_frac, resign_consecutive, resign_min_ply, gamma);
}

// One keep-best GATE game between two nets. white_net plays the White plies,
// black_net the Black plies.
struct MatchGame {
    std::string outcome;             // "white" / "black" / "draw" (winner perspective)
    std::vector<std::string> moves;  // ucis played, in order (for parity testing)
};

// Play one gate game from `start_fen`. Pure mirror of fleet_arena._play_from driven
// by _native_picker: per-move FRESH-tree native PUCT (NO tree reuse — the 2-net gate
// descends two plies between a side's turns, so single-ply reuse never applies),
// argmax-visit pick (the run_mcts_native `chosen`; NOT temperature-sampled), and a
// per-move-incrementing Dirichlet seed (`dir_seed_base + ply`) so repeated gate games
// of the same unit diverge (light noise — play stays strong). No py:: → GIL-free.
inline MatchGame play_match_game(const ChesskersNet& white_net, const ChesskersNet& black_net,
                                 const std::string& start_fen, int n_sims, double c_puct,
                                 double dirichlet_alpha, double dirichlet_eps, int max_plies,
                                 uint64_t dir_seed_base, double gamma) {
    Board board = parse_fen(start_fen);
    MatchGame g;
    std::string final_status, final_winner;
    int ply = 0;
    while (ply < max_plies) {
        const auto st = detect_status(board);
        if (!st.status.empty()) {
            final_status = st.status;
            final_winner = st.winner;
            break;
        }
        auto legal = gen_legal_native(board);
        if (legal.empty()) break;  // no moves, not flagged terminal — end as draw
        const ChesskersNet& net = board.turn_white ? white_net : black_net;
        DirectEvaluator ev(net);  // gate play stays serial (per-move fresh tree)
        auto root = std::make_unique<PuctNode>();  // fresh tree per move
        root->board = board;
        std::mt19937_64 rng(dir_seed_base + (uint64_t)ply);
        expand_dirichlet_search_pure(root.get(), net, ev, n_sims, c_puct, gamma, dirichlet_alpha,
                                     dirichlet_eps, rng);
        // argmax visit (first max in child/gen order) — the run_mcts_native `chosen`.
        std::string chosen;
        int best = -1;
        for (auto& c : root->children)
            if (c->visits > best) {
                best = c->visits;
                chosen = c->uci;
            }
        if (chosen.empty()) break;  // no children — picker None -> draw
        int idx = -1;
        for (int i = 0; i < (int)legal.size(); ++i)
            if (legal[i].uci == chosen) {
                idx = i;
                break;
            }
        if (idx < 0) break;  // chosen not in legal (can't happen) -> draw, like _play_from
        g.moves.push_back(chosen);
        board = apply_native(board, legal[idx]);
        ++ply;
    }
    g.outcome = outcome_from_pure(final_status, final_winner);
    return g;
}

// Fan `num_games` pure games across `num_threads` worker threads. Game i is
// seeded `base_seed + i`, so the result is independent of the thread count and
// equals a single-threaded play_game_pure(seed=base_seed+i). The net is shared
// read-only (ChesskersNet::eval is const). Call with the GIL released.
inline std::vector<PureGame> play_games_pure(const Board& start, const ChesskersNet& net,
                                             int num_games, int num_threads, int n_sims,
                                             double c_puct, double temperature, int temp_cutoff_plies,
                                             int max_plies, double dirichlet_alpha,
                                             double dirichlet_eps, uint64_t base_seed,
                                             double resign_threshold, double resign_no_resign_frac,
                                             int resign_consecutive, int resign_min_ply,
                                             double gamma) {
    std::vector<PureGame> games(std::max(0, num_games));
    if (num_games <= 0) return games;
    const int nthreads = std::max(1, std::min(num_threads, num_games));
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= num_games) break;
            games[i] = play_game_pure(start, net, n_sims, c_puct, temperature, temp_cutoff_plies,
                                      max_plies, dirichlet_alpha, dirichlet_eps, base_seed + (uint64_t)i,
                                      resign_threshold, resign_no_resign_frac, resign_consecutive,
                                      resign_min_ply, gamma);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    return games;
}

// Phase 6d — GPU leaf batching (the lc0 design). Fan `num_games` across `batch_size`
// concurrent game threads, ALL sharing one BatchEvaluator: every running game submits
// its current leaf, the gatherer runs ONE batched forward over them (`forward`, the GPU
// path), and the games resume. Each game still runs an ordinary sequential search, so
// game i is BYTE-IDENTICAL to play_game_pure(seed=base_seed+i) whenever `forward` is
// byte-identical to serial eval (CPU eval_batch is; the Metal forward is ~1e-4-close).
// The win is GPU: one forward over `batch_size` leaves instead of `batch_size` small
// forwards. Call with the GIL released.
inline std::vector<PureGame> play_games_batched_pure(
    const Board& start, const ChesskersNet& net, int num_games, int batch_size, int n_sims,
    double c_puct, double temperature, int temp_cutoff_plies, int max_plies, double dirichlet_alpha,
    double dirichlet_eps, uint64_t base_seed, double resign_threshold, double resign_no_resign_frac,
    int resign_consecutive, int resign_min_ply, double gamma, TrunkForwardFn trunk_forward,
    int concurrency = 0) {
    std::vector<PureGame> games(std::max(0, num_games));
    if (num_games <= 0) return games;
    // concurrency = number of game threads; batch_size = the GPU batch WIDTH (gatherer cap).
    // concurrency>batch_size oversubscribes (lc0 pipelining): while a batch is on the GPU the
    // extra threads compute their next leaves, so the GPU never waits on leaf production. The
    // gatherer fires at `pending>=cap`, so a full batch is always ready. concurrency<=0 => no
    // oversubscription (threads==cap). Game results are unchanged — pure inference transport.
    const int cap = std::max(1, std::min(batch_size, num_games));
    const int want = concurrency > 0 ? concurrency : cap;
    const int nworkers = std::max(1, std::min(want, num_games));

    BatchEvaluator bev(net, std::move(trunk_forward), cap, nworkers);
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= num_games) {
                bev.worker_exit();  // exhausted: stop the gatherer waiting on this slot
                break;
            }
            games[i] = play_game_pure(start, net, bev, n_sims, c_puct, temperature,
                                      temp_cutoff_plies, max_plies, dirichlet_alpha, dirichlet_eps,
                                      base_seed + (uint64_t)i, resign_threshold,
                                      resign_no_resign_frac, resign_consecutive, resign_min_ply,
                                      gamma);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nworkers);
    for (int t = 0; t < nworkers; ++t) pool.emplace_back(worker);
    bev.run_gatherer();  // drives the batched forwards on THIS thread until all workers exit
    for (auto& th : pool) th.join();
    return games;
}

}  // namespace cc
