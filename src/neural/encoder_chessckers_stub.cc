/*
  This file is part of akshay-chessckers-0 (Chessckers port of Leela Chess Zero).

  Stub implementations of lc0's chess encoder/decoder API (encoder.h). The real
  lc0 encoder.cc/decoder.cc (fixed-1858 chess planes + move<->index map) are
  excluded from the Chessckers build; the chessckers backend does its own encoding
  via cc::encode.hpp. These stubs exist only so the few remaining in-build callers
  (trainingdata.cc, reader.cc, and a debug line in search/classic/search.cc) link.

  They return safe defaults and are NOT exercised by UCI play / search (which use
  the chessckers backend's EvalResult directly). Chessckers training data (Phase 5)
  will replace the trainingdata path with the cc::chunk format, at which point these
  stubs go away.
*/

#include "neural/encoder.h"

namespace lczero {

int TransformForPosition(pblczero::NetworkFormat::InputFormat,
                         const PositionHistory&) {
  return 0;
}

InputPlanes EncodePositionForNN(pblczero::NetworkFormat::InputFormat,
                                const PositionHistory&, int, FillEmptyHistory,
                                int* transform_out) {
  if (transform_out) *transform_out = 0;
  return {};
}

InputPlanes EncodePositionForNN(pblczero::NetworkFormat::InputFormat,
                                std::span<const Position>, int, FillEmptyHistory,
                                int* transform_out) {
  if (transform_out) *transform_out = 0;
  return {};
}

bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat) { return false; }
bool IsCanonicalArmageddonFormat(pblczero::NetworkFormat::InputFormat) {
  return false;
}
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat) { return false; }
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat) { return false; }

uint16_t MoveToNNIndex(Move, int) { return 0; }
Move MoveFromNNIndex(int, int) { return Move(); }

}  // namespace lczero
