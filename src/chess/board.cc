/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

// Chessckers port: ChessBoard + Move adapter bodies over the cc:: reference core.
// This TU is the single place that pulls the heavy cc:: headers (movegen/apply/nn);
// types.h/board.h only forward-declare or use the lightweight cc::Board struct.

#include "chess/board.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "chessckers/apply.hpp"          // cc::detect_status, apply_*
#include "chessckers/movegen.hpp"        // all_black_legal_moves (via native_move)
#include "chessckers/movegen_white.hpp"  // white_in_chessckers_check
#include "chessckers/native_move.hpp"    // cc::NativeMove, gen_legal_native, apply_native

namespace lczero {

namespace {
// from/to square of a cc::NativeMove, in lc0's 0..63 indexing (a1=0..h8=63).
int NativeFromSq(const cc::NativeMove& nm) {
  if (nm.is_white) return nm.white_src.from_sq;
  int sq = 0;
  std::visit([&](auto&& x) { sq = cc::parse_square(x.from_name); }, nm.black_src);
  return sq;
}
int NativeToSq(const cc::NativeMove& nm) {
  if (nm.is_white) {
    return nm.is_castling_alt ? nm.white_src.castling_rook_sq : nm.white_src.to_sq;
  }
  int sq = 0;
  std::visit([&](auto&& x) { sq = cc::parse_square(x.to_name); }, nm.black_src);
  return sq;
}
}  // namespace

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

Move Move::White(Square from, Square to) {
  auto nm = std::make_shared<cc::NativeMove>();
  nm->is_white = true;
  nm->uci = from.ToString() + to.ToString();
  nm->white_src.from_sq = from.as_idx();
  nm->white_src.to_sq = to.as_idx();
  return Move(std::move(nm));
}

// Chess-only compat shims (pgn.h SAN parsing). Compile-only: the produced moves
// are NOT valid Chessckers apply payloads and must not reach cc::apply_native.
Move Move::WhitePromotion(Square from, Square to, PieceType promotion_piece) {
  auto nm = std::make_shared<cc::NativeMove>();
  nm->is_white = true;
  nm->uci = from.ToString() + to.ToString() + promotion_piece.ToString(false);
  nm->white_src.from_sq = from.as_idx();
  nm->white_src.to_sq = to.as_idx();
  return Move(std::move(nm));
}
Move Move::WhiteCastling(File king, File rook) {
  auto nm = std::make_shared<cc::NativeMove>();
  nm->is_white = true;
  nm->uci = Square(king, kRank1).ToString() + Square(rook, kRank1).ToString();
  nm->white_src.from_sq = Square(king, kRank1).as_idx();
  nm->white_src.to_sq = Square(rook, kRank1).as_idx();
  return Move(std::move(nm));
}
Move Move::WhiteEnPassant(Square from, Square to) {
  auto nm = std::make_shared<cc::NativeMove>();
  nm->is_white = true;
  nm->uci = from.ToString() + to.ToString();
  nm->white_src.from_sq = from.as_idx();
  nm->white_src.to_sq = to.as_idx();
  return Move(std::move(nm));
}

bool Move::operator==(const Move& other) const {
  if (!nm_ || !other.nm_) return nm_.get() == other.nm_.get();
  return nm_->uci == other.nm_->uci &&
         nm_->is_castling_alt == other.nm_->is_castling_alt;
}

std::string Move::ToString(bool /*is_chess960*/) const {
  return nm_ ? nm_->uci : "0000";
}

Square Move::from() const {
  return nm_ ? Square::FromIdx(NativeFromSq(*nm_)) : Square();
}
Square Move::to() const {
  return nm_ ? Square::FromIdx(NativeToSq(*nm_)) : Square();
}

// ---------------------------------------------------------------------------
// ChessBoard
// ---------------------------------------------------------------------------

// version_5 endgame: Black 2-King towers on d8 & e8 vs White's lone king (e1),
// Black to move. A long KK-vs-K win (no longer a 1-move mate) — Black must march
// the towers down the board and deliver the charge mate, so the net has to LEARN
// the conversion instead of search solving it in one ply. No opening double-move.
const char* ChessBoard::kStartposFen =
    "3kk3/8/8/8/8/8/8/4K3"
    "[d8:kk,e8:kk] b - - 0 1";
const BitBoard ChessBoard::kPawnMask = BitBoard(0);  // unused (no en passant)

void ChessBoard::SetFromFen(std::string_view fen, int* rule50_ply, int* moves) {
  b_ = cc::parse_fen(std::string(fen));
  if (rule50_ply) *rule50_ply = b_.halfmove;
  // Half-move (ply) count since game start, lc0 convention.
  if (moves) *moves = (b_.fullmove - 1) * 2 + (b_.turn_white ? 0 : 1);
}

void ChessBoard::Clear() { b_ = cc::Board{}; }

MoveList ChessBoard::GenerateLegalMoves() const {
  MoveList out;
  for (auto& nm : cc::gen_legal_native(b_)) {
    out.push_back(Move(std::make_shared<cc::NativeMove>(std::move(nm))));
  }
  return out;
}

bool ChessBoard::ApplyMove(Move move) {
  const auto& nm = move.native();
  b_ = cc::apply_native(b_, *nm);
  return b_.halfmove == 0;  // rule50 reset signal (cosmetic in Chessckers)
}

bool ChessBoard::IsUnderCheck() const {
  if (!b_.turn_white) return false;  // "check" only applies to White (chess side)
  const uint64_t wk = b_.kings & b_.occupied_white;
  if (!wk) return true;  // king already captured
  const int wk_sq = __builtin_ctzll(wk);
  return cc::white_in_chessckers_check(b_.occupied(), b_.occupied_white, wk_sq,
                                       b_.stacks);
}

Move ChessBoard::ParseMove(std::string_view move_str) const {
  for (auto& nm : cc::gen_legal_native(b_)) {
    if (nm.uci == move_str) {
      return Move(std::make_shared<cc::NativeMove>(std::move(nm)));
    }
  }
  return Move();  // null move if not found
}

uint64_t ChessBoard::Hash() const {
  uint64_t stack_hash = 0;
  for (const auto& [sq, pieces] : b_.stacks) {
    uint64_t h = sq;
    for (char c : pieces) h = h * 131 + static_cast<unsigned char>(c);
    stack_hash = HashCat(stack_hash, h);
  }
  const uint64_t flags =
      (static_cast<uint64_t>(b_.castling_rights)) ^
      (static_cast<uint64_t>(b_.turn_white) << 1) ^
      (static_cast<uint64_t>(b_.white_moves_left) << 2) ^
      (static_cast<uint64_t>(b_.rank8_count) << 5) ^
      (static_cast<uint64_t>(b_.ep_square + 1) << 8);
  return HashCat({b_.pawns, b_.knights, b_.bishops, b_.rooks, b_.queens,
                  b_.kings, b_.occupied_white, b_.occupied_black, flags,
                  stack_hash});
}

bool ChessBoard::operator==(const ChessBoard& o) const {
  // Position identity (ignores half/full-move counters, like lc0's operator==).
  return b_.pawns == o.b_.pawns && b_.knights == o.b_.knights &&
         b_.bishops == o.b_.bishops && b_.rooks == o.b_.rooks &&
         b_.queens == o.b_.queens && b_.kings == o.b_.kings &&
         b_.occupied_white == o.b_.occupied_white &&
         b_.occupied_black == o.b_.occupied_black &&
         b_.castling_rights == o.b_.castling_rights &&
         b_.ep_square == o.b_.ep_square && b_.turn_white == o.b_.turn_white &&
         b_.white_moves_left == o.b_.white_moves_left &&
         b_.rank8_count == o.b_.rank8_count && b_.stacks == o.b_.stacks;
}

std::string ChessBoard::DebugString() const { return cc::serialize_fen(b_); }

const ChessBoard ChessBoard::kStartposBoard(ChessBoard::kStartposFen);

std::string BoardToFen(const ChessBoard& board) {
  return cc::serialize_fen(board.cc());
}

}  // namespace lczero
