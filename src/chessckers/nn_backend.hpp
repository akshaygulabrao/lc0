// Chessckers C++ engine — pluggable NN backend registry (lc0 NetworkFactory analog).
//
// lc0 selects an inference backend (cuda/metal/blas/...) by NAME from a registry, or AUTO-picks
// the highest-priority one that successfully instantiates on the box. We mirror that, but the
// backend-variable op here is just the BATCHED TRUNK forward (K positions -> K [c_filters*100]
// feature maps): the trunk is the fixed-shape, batchable, GPU-friendly part, while the
// variable-length per-move gather head stays on the CPU game threads (Phase 6e) for EVERY
// backend — so a backend never has to deal with our non-lc0 policy head. That keeps the seam
// exactly the TrunkForwardFn the batched self-play driver already consumes (selfplay.hpp); the
// search/self-play layer never changes when a backend is added.
//
// Two builtins: `cpu` (BLAS trunk_v2_batch, priority 0, always available — the auto fallback)
// and `metal` (MPSGraph MetalTrunkV2, priority 100, Apple-only, make() returns nullptr when no
// Metal device so auto falls through to cpu). To ADD a backend (e.g. cuda): build its source
// under a CC_HAVE_CUDA define (see CMakeLists), then add ONE registration line to the registry
// ctor below, guarded by #ifdef CC_HAVE_CUDA. Nothing else moves.
#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn.hpp"
#ifdef CC_HAVE_METAL
#include "nn_metal.h"
#endif

namespace cc {

using BackendOpts = std::map<std::string, std::string>;

// A Backend is a ready-to-run form of the net's TRUNK for one device (lc0's Network).
struct Backend {
    virtual ~Backend() = default;
    // K positions (each flat NCHW [c_in*100]) -> K trunk feature maps [c_filters*100].
    virtual std::vector<std::vector<float>> trunk_batch(
        const std::vector<std::vector<float>>& positions) = 0;
};

// make() returns nullptr when this backend can't run on this box (no device / unsupported arch /
// disabled at build) so the auto-picker falls through to the next-highest priority.
using BackendFactory =
    std::function<std::unique_ptr<Backend>(const ChesskersNet&, const BackendOpts&)>;

struct BackendInfo {
    std::string name;
    int priority;  // auto-pick = highest priority whose make() succeeds; cpu==0 is the floor
    BackendFactory make;
};

// ---- builtins ----
struct CpuTrunkBackend : Backend {
    const ChesskersNet& net;
    explicit CpuTrunkBackend(const ChesskersNet& n) : net(n) {}
    std::vector<std::vector<float>> trunk_batch(
        const std::vector<std::vector<float>>& positions) override {
        return net.trunk_v2_batch(positions);
    }
};

#ifdef CC_HAVE_METAL
struct MetalTrunkBackend : Backend {
    std::shared_ptr<MetalTrunkV2> trunk;  // owns the cached MPSGraph; `net` must outlive it
    explicit MetalTrunkBackend(std::shared_ptr<MetalTrunkV2> t) : trunk(std::move(t)) {}
    std::vector<std::vector<float>> trunk_batch(
        const std::vector<std::vector<float>>& positions) override {
        return trunk->run(positions);
    }
};
#endif

// Registry — a Meyers singleton (its static-local init is once + thread-safe). Builtins are
// registered in the ctor; add() is idempotent by name so a future header-defined registration
// included in multiple TUs still registers once.
class BackendRegistry {
  public:
    static BackendRegistry& get() {
        static BackendRegistry inst;
        return inst;
    }

    void add(BackendInfo info) {
        for (const auto& e : infos_)
            if (e.name == info.name) return;  // idempotent (multi-TU header-include safe)
        infos_.push_back(std::move(info));
    }

    std::vector<std::string> available() const {
        std::vector<std::string> names;
        for (const auto& e : infos_) names.push_back(e.name);
        return names;
    }

    // name=="" / "auto" -> the highest-priority backend whose make() succeeds (cpu is the floor).
    // explicit name -> that backend, or throw if unknown / unavailable on this box (lc0 behaviour:
    // when you NAME a backend you want to know it didn't load, not silently get another).
    std::unique_ptr<Backend> create(const ChesskersNet& net, const std::string& name,
                                    const BackendOpts& opts = {}) const {
        std::vector<const BackendInfo*> order;
        order.reserve(infos_.size());
        for (const auto& e : infos_) order.push_back(&e);
        std::sort(order.begin(), order.end(), [](const BackendInfo* a, const BackendInfo* b) {
            return a->priority > b->priority;
        });
        if (name.empty() || name == "auto") {
            for (const auto* e : order)
                if (auto b = e->make(net, opts)) return b;
            throw std::runtime_error("no NN backend available (cpu builtin missing?)");
        }
        for (const auto* e : order)
            if (e->name == name) {
                if (auto b = e->make(net, opts)) return b;
                throw std::runtime_error("NN backend '" + name + "' is unavailable on this machine");
            }
        throw std::runtime_error("unknown NN backend '" + name + "'");
    }

  private:
    BackendRegistry() {
        add({"cpu", 0, [](const ChesskersNet& net, const BackendOpts&) -> std::unique_ptr<Backend> {
                 return std::make_unique<CpuTrunkBackend>(net);
             }});
#ifdef CC_HAVE_METAL
        add({"metal", 100,
             [](const ChesskersNet& net, const BackendOpts&) -> std::unique_ptr<Backend> {
                 auto t = std::make_shared<MetalTrunkV2>(net);
                 if (!t->ok()) return nullptr;  // no Metal device -> auto falls through to cpu
                 return std::make_unique<MetalTrunkBackend>(std::move(t));
             }});
#endif
        // To add a backend, e.g. CUDA:
        // #ifdef CC_HAVE_CUDA
        //     add({"cuda", 100, [](const ChesskersNet& net, const BackendOpts& o) { ... }});
        // #endif
    }
    std::vector<BackendInfo> infos_;
};

}  // namespace cc
