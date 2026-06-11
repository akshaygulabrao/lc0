// Head-isolation parity: feed the SAME trunk feature map F to the CPU oracle
// (ChesskersNet::value_v2 / policy_logits_v2) AND the GPU heads
// (MetalTrunkV2::eval_heads_from_F), so any divergence is the GPU head port — not the
// trunk's pre-existing cuBLAS/MPSGraph-vs-CBLAS float drift. F comes from the GPU trunk
// (metal.run) and is handed, bit-identical, to both paths.
//
// Build (from repo root), passing a net .bin. Match the meson flags for nn_metal.mm
// (CC_HAVE_METAL, NO -fobjc-arc — the graph owns its tensors):
//   clang++ -std=c++20 -O2 -Isrc -Isrc/chessckers -DACCELERATE_NEW_LAPACK -DCC_HAVE_METAL \
//     tests_chessckers/head_parity_test.cc src/chessckers/nn_metal.mm \
//     -framework Accelerate -framework Foundation -framework Metal \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -o /tmp/cc_head_parity && \
//   /tmp/cc_head_parity ../chessckers/engine/net-<hash>.bin

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "chessckers/board.hpp"
#include "chessckers/encode.hpp"
#include "chessckers/native_move.hpp"
#include "chessckers/nn.hpp"
#if defined(CC_HAVE_CUDA)
#include "chessckers/nn_cuda.h"
#else
#include "chessckers/nn_metal.h"
#endif

using namespace cc;

#if defined(CC_HAVE_CUDA)
using Trunk = CudaTrunkV2;
static constexpr const char* kBackend = "CUDA";
#else
using Trunk = MetalTrunkV2;
static constexpr const char* kBackend = "Metal";
#endif

static int g_fail = 0;

static void TestFen(const ChesskersNet& net, Trunk& metal, const std::string& fen) {
  Board b = parse_fen(fen);
  auto legal = gen_legal_native(b);
  std::vector<float> pos = encode_position_v2(b);
  std::vector<std::vector<float>> moves;
  moves.reserve(legal.size());
  for (const auto& m : legal) moves.push_back(encode_native_move(net, m));

  // One F from the GPU trunk, handed to BOTH paths.
  auto Fs = metal.run({pos});
  if (Fs.empty()) { std::printf("  (no Metal GPU) skip: %s\n", fen.c_str()); return; }
  const auto& F = Fs[0];

  // CPU oracle on F.
  const float cpu_v = net.value_v2(F);
  const auto cpu_logits = net.policy_logits_v2(F, moves);
  const auto cpu_priors = softmax_priors(cpu_logits.data(), (int)cpu_logits.size());

  // GPU heads on the SAME F.
  auto gpu = metal.eval_heads_from_F({F}, {moves});
  const float gpu_v = gpu[0].first;
  const auto& gpu_priors = gpu[0].second;

  const float dv = std::fabs(gpu_v - cpu_v);
  float dp = 0.0f;
  const bool same_n = gpu_priors.size() == cpu_priors.size();
  if (same_n)
    for (size_t i = 0; i < cpu_priors.size(); ++i)
      dp = std::max(dp, std::fabs(gpu_priors[i] - cpu_priors[i]));

  const bool ok_v = dv < 1e-3f;
  const bool ok_p = same_n && dp < 1e-2f;
  if (!ok_v || !ok_p) ++g_fail;
  std::printf("  N=%-3zu |dv|=%.2e %-4s max|dp|=%.2e %-4s  (cpu_v=%+.4f gpu_v=%+.4f)  %s\n",
              moves.size(), dv, ok_v ? "OK" : "FAIL", dp, ok_p ? "OK" : "FAIL", cpu_v, gpu_v,
              fen.substr(0, 40).c_str());
}

// Batched (K>1) parity: ALL boards in ONE eval — exercises the flattened-M policy head
// across boards with different move counts (board_off scatter, per-board softmax slices,
// gfrom=board*100+from across boards). This is the real production scenario.
static void TestBatch(const ChesskersNet& net, Trunk& metal,
                      const std::vector<std::string>& fens) {
  std::vector<std::vector<float>> positions;
  std::vector<std::vector<std::vector<float>>> moves_per;
  for (const auto& fen : fens) {
    Board b = parse_fen(fen);
    positions.push_back(encode_position_v2(b));
    std::vector<std::vector<float>> mv;
    for (const auto& m : gen_legal_native(b)) mv.push_back(encode_native_move(net, m));
    moves_per.push_back(std::move(mv));
  }
  auto Fs = metal.run(positions);
  if (Fs.empty()) { std::printf("  (no Metal GPU) skip batch\n"); return; }
  auto gpu = metal.eval_heads_from_F(Fs, moves_per);  // K boards in one call
  std::printf("BATCHED (K=%zu in one eval):\n", fens.size());
  for (size_t k = 0; k < fens.size(); ++k) {
    const float cpu_v = net.value_v2(Fs[k]);
    const auto cpu_logits = net.policy_logits_v2(Fs[k], moves_per[k]);
    const auto cpu_priors = softmax_priors(cpu_logits.data(), (int)cpu_logits.size());
    const float dv = std::fabs(gpu[k].first - cpu_v);
    float dp = 0.0f;
    const bool same_n = gpu[k].second.size() == cpu_priors.size();
    if (same_n)
      for (size_t i = 0; i < cpu_priors.size(); ++i)
        dp = std::max(dp, std::fabs(gpu[k].second[i] - cpu_priors[i]));
    const bool ok = (dv < 1e-3f) && same_n && (dp < 1e-2f);
    if (!ok) ++g_fail;
    std::printf("  board %zu: N=%-3zu |dv|=%.2e max|dp|=%.2e %s\n", k, cpu_priors.size(), dv, dp,
                ok ? "OK" : "FAIL");
  }
}

// Moves-left head: confirm has_moves_left detection, that eval_batch's m_out is populated on
// BOTH the CPU oracle and the GPU path, and that the GPU path's m_out is EXACTLY the host
// moves_left_v2 of the GPU trunk's F (isolates the m_out wiring from trunk float drift — the
// CPU-vs-GPU absolute values legitimately differ by the same trunk drift the heads see). Also a
// sanity gate: values must be finite and >= 0 (Softplus output, expected plies-to-end).
static void TestMovesLeft(const ChesskersNet& net, Trunk& metal,
                          const std::vector<std::string>& fens) {
  std::printf("MOVES-LEFT (has_moves_left=%d):\n", net.has_moves_left);
  if (!net.has_moves_left) { std::printf("  net has no moves-left head; skip\n"); return; }
  std::vector<std::vector<float>> positions;
  std::vector<std::vector<std::vector<float>>> moves_per;
  for (const auto& fen : fens) {
    Board b = parse_fen(fen);
    positions.push_back(encode_position_v2(b));
    std::vector<std::vector<float>> mv;
    for (const auto& m : gen_legal_native(b)) mv.push_back(encode_native_move(net, m));
    moves_per.push_back(std::move(mv));
  }
  std::vector<float> m_cpu, m_gpu;
  net.eval_batch(positions, moves_per, &m_cpu);    // CPU trunk + CPU moves-left head
  metal.eval_batch(positions, moves_per, &m_gpu);  // GPU trunk + host moves-left head
  auto Fs = metal.run(positions);
  for (size_t k = 0; k < fens.size(); ++k) {
    const float ref = net.moves_left_v2(Fs[k]);  // what the GPU path must reproduce exactly
    const float dwire = (k < m_gpu.size()) ? std::fabs(m_gpu[k] - ref) : 9.9f;
    const bool finite_pos = k < m_cpu.size() && std::isfinite(m_cpu[k]) && m_cpu[k] >= 0.0f &&
                            k < m_gpu.size() && std::isfinite(m_gpu[k]) && m_gpu[k] >= 0.0f;
    const bool wire_ok = dwire < 1e-4f;
    if (!finite_pos || !wire_ok) ++g_fail;
    std::printf("  board %zu: ml_cpu=%7.2f ml_gpu=%7.2f wire|d|=%.1e %-3s %-9s  %s\n", k,
                m_cpu[k], (k < m_gpu.size() ? m_gpu[k] : -1.f), dwire, finite_pos ? "pos" : "BAD",
                wire_ok ? "wire-OK" : "WIRE-FAIL", fens[k].substr(0, 32).c_str());
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: %s net.bin\n", argv[0]); return 2; }
  ChesskersNet net(argv[1]);
  std::printf("net is_v2=%d c_filters=%d d_hidden=%d d_move=%d\n", net.is_v2, net.c_filters,
              net.d_hidden, net.d_move);
  Trunk metal(net);
  std::printf("backend: %s\n", kBackend);
  if (!metal.ok()) { std::printf("%s trunk not ok (no GPU?)\n", kBackend); return 1; }

  const std::vector<std::string> fens = {
      // canonical opening (White, double-move) — value + White policy
      "pppppppp/pkkkkkkp/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR"
      "[a6:s,b6:s,c6:s,d6:s,e6:s,f6:s,g6:s,h6:s,a7:s,b7:k,c7:k,d7:k,e7:k,f7:k,g7:k,h7:s,"
      "a8:s,b8:s,c8:s,d8:s,e8:s,f8:s,g8:s,h8:s] w KQkq - 0 1 {wm:2}",
      // reduced, Black to move — diagonal moves
      "8/8/3kkk2/8/8/8/PPPPPPPP/4K3[d6:kk,e6:kk,f6:kk] b - - 0 1",
      // Black tower 2 diagonal squares from a White pawn — should force a capture (path mask)
      "8/8/2k5/8/P7/8/8/4K3[c6:kk] b - - 0 1",
  };
  for (const auto& f : fens) TestFen(net, metal, f);
  TestBatch(net, metal, fens);
  TestMovesLeft(net, metal, fens);
  std::printf("%s\n", g_fail ? "HEAD PARITY FAIL" : "HEAD PARITY OK");
  return g_fail ? 1 : 0;
}
