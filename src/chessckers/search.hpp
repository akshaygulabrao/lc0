// Chessckers C++ engine — Slice 5b: native PUCT tree.
//
// Pure tree math (node, selection, backup) — a 1:1 port of mcts_puct.py's
// _puct_score / _select_child / _backup / _select_to_leaf and the Lc0 PUCT
// refinements (cpuct grows with parent visits; FPU reduction on unvisited
// children). The leaf NN evaluation crosses into Python (slice 5c, in
// bindings.cpp); everything here stays native.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "board.hpp"

namespace cc {

constexpr double TERMINAL_LOSS_VALUE = -1.0;  // side-to-move at a terminal has just lost
constexpr double TERMINAL_DRAW_VALUE = 0.0;   // stalemate
constexpr double CPUCT_FACTOR = 2.0;
constexpr double CPUCT_BASE = 19652.0;
constexpr double FPU_REDUCTION = 0.25;

struct PuctNode {
    Board board;
    std::string uci;              // move that led here ("" for the root)
    double prior = 0.0;           // P(this move | parent)
    int visits = 0;
    double total_value = 0.0;
    bool expanded = false;
    bool is_terminal = false;
    std::string terminal_status;  // "" == None
    std::vector<std::unique_ptr<PuctNode>> children;

    double q() const { return visits > 0 ? total_value / visits : 0.0; }
};

inline double terminal_value(const PuctNode& n) {
    return n.terminal_status == "stalemate" ? TERMINAL_DRAW_VALUE : TERMINAL_LOSS_VALUE;
}

// _select_child: argmax PUCT over children. cpuct grows with parent visits; an
// unvisited child is valued at the FPU (parent Q reduced by explored policy mass).
// Ties resolve to the first child in legal-move order (matches Python's max()).
inline PuctNode* select_child(PuctNode& parent, double c_puct) {
    const double pv = parent.visits;
    const double cpuct = c_puct + CPUCT_FACTOR * std::log((pv + CPUCT_BASE) / CPUCT_BASE);
    double visited_prior = 0.0;
    for (auto& c : parent.children)
        if (c->visits > 0) visited_prior += c->prior;
    const double fpu = parent.q() - FPU_REDUCTION * std::sqrt(std::max(visited_prior, 0.0));

    PuctNode* best = nullptr;
    double best_score = -1e300;
    for (auto& c : parent.children) {
        const double q = c->visits > 0 ? -c->q() : fpu;
        const double u = cpuct * c->prior * std::sqrt(std::max(pv, 1.0)) / (1.0 + c->visits);
        const double s = q + u;
        if (s > best_score) {
            best_score = s;
            best = c.get();
        }
    }
    return best;
}

inline std::vector<PuctNode*> select_to_leaf(PuctNode* root, double c_puct) {
    std::vector<PuctNode*> path{root};
    PuctNode* node = root;
    while (node->expanded && !node->is_terminal && !node->children.empty()) {
        node = select_child(*node, c_puct);
        path.push_back(node);
    }
    return path;
}

// Negamax backup with optional per-ply value discount (gamma; 1.0 == off).
inline void backup(const std::vector<PuctNode*>& path, double value, double gamma) {
    double sign = 1.0, discount = 1.0;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        (*it)->visits += 1;
        (*it)->total_value += sign * discount * value;
        sign = -sign;
        discount *= gamma;
    }
}

}  // namespace cc
