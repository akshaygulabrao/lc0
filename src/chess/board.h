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

// Chessckers port: ChessBoard is an adapter over the reference engine's cc::Board
// (10x10-path checkers-vs-chess hybrid). It keeps lc0's ChessBoard public API so
// the generic search/uci/engine layers compile unchanged, but delegates rules to
// the cc:: core (chess/board.cc). See PORTING.md.
//
// Gotcha #1: Chessckers is asymmetric -> NO board mirror. Mirror() is a no-op and
// the board is ALWAYS in the absolute (White-frame) coordinates. flipped() means
// "Black to move" without anything having been mirrored. ours()/theirs() are
// therefore White/Black absolutely, not side-to-move-relative.

#pragma once

#include <cassert>
#include <string>

#include "chess/bitboard.h"
#include "chess/types.h"
#include "chessckers/board.hpp"  // cc::Board (lightweight: no BLAS/nn pulled in)
#include "utils/hashcat.h"

namespace lczero {

// Kept for API parity with lc0's legal-move detection. In Chessckers, legality is
// fully resolved inside cc:: move generation, so the search never relies on this.
class KingAttackInfo {
 public:
  bool in_check() const { return attack_lines_.as_int(); }
  bool in_double_check() const { return double_check_; }
  bool is_pinned(const Square square) const {
    return pinned_pieces_.get(square);
  }
  bool is_on_attack_line(const Square square) const {
    return attack_lines_.get(square);
  }

  bool double_check_ = 0;
  BitBoard pinned_pieces_ = {0};
  BitBoard attack_lines_ = {0};
};

// Represents a Chessckers position (board + stacks + turn/win state).
class ChessBoard {
 public:
  ChessBoard() = default;
  ChessBoard(const ChessBoard&) = default;
  ChessBoard(const std::string& fen) { SetFromFen(fen); }

  ChessBoard& operator=(const ChessBoard&) = default;

  static const char* kStartposFen;
  static const ChessBoard kStartposBoard;
  static const BitBoard kPawnMask;

  // Sets position from (extended) FEN. Fills rule50/move counters if requested.
  void SetFromFen(std::string_view fen, int* rule50_ply = nullptr,
                  int* moves = nullptr);
  void Clear();
  // No-op in Chessckers (asymmetric variant; see gotcha #1). Kept for API parity.
  void Mirror() {}

  // Generates legal moves (cc:: gen is already fully legal, so pseudolegal==legal).
  MoveList GeneratePseudolegalMoves() const { return GenerateLegalMoves(); }
  MoveList GenerateLegalMoves() const;
  // Applies a (legal) move. Returns true if the rule50 counter should be reset.
  bool ApplyMove(Move move);
  // Whether White's king is in Chessckers-check (only meaningful when White to move).
  bool IsUnderCheck() const;
  // Chessckers has no insufficient-material draw.
  bool HasMatingMaterial() const { return true; }
  // cc:: moves are pre-validated, so every generated move is legal.
  bool IsLegalMove(Move, const KingAttackInfo&) const { return true; }
  KingAttackInfo GenerateKingAttackInfo() const { return {}; }
  // Syzygy-only in lc0; unused in Chessckers (tablebases excluded). Best-effort.
  bool IsUnderAttack(Square) const { return false; }

  // Resolves a UCI string to a legal Move (matches against generated moves).
  Move ParseMove(std::string_view move_str) const;

  uint64_t Hash() const;

  // cc:: castling_rights uses python-chess rook-corner bits:
  //   a1(Q)=bit0, h1(K)=bit7, a8(q)=bit56, h8(k)=bit63.
  class Castlings {
   public:
    Castlings() = default;
    explicit Castlings(uint64_t cc_rights) {
      if (cc_rights & (1ULL << 7)) data_ |= 1;    // White kingside  (K)
      if (cc_rights & (1ULL << 0)) data_ |= 2;    // White queenside (Q)
      if (cc_rights & (1ULL << 63)) data_ |= 4;   // Black kingside  (k)
      if (cc_rights & (1ULL << 56)) data_ |= 8;   // Black queenside (q)
    }

    bool we_can_00() const { return data_ & 1; }
    bool we_can_000() const { return data_ & 2; }
    bool they_can_00() const { return data_ & 4; }
    bool they_can_000() const { return data_ & 8; }
    bool no_legal_castle() const { return data_ == 0; }
    uint8_t as_int() const { return data_; }
    bool operator==(const Castlings& other) const = default;

    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string r;
      if (we_can_00()) r += 'K';
      if (we_can_000()) r += 'Q';
      if (they_can_00()) r += 'k';
      if (they_can_000()) r += 'q';
      return r;
    }
    std::string DebugString() const { return as_string(); }

    File our_queenside_rook{kFileA};
    File their_queenside_rook{kFileA};
    File our_kingside_rook{kFileH};
    File their_kingside_rook{kFileH};

   private:
    uint8_t data_ = 0;
  };

  std::string DebugString() const;

  // Absolute-frame bitboard accessors (gotcha #1: ours=White, theirs=Black).
  BitBoard ours() const { return BitBoard(b_.occupied_white); }
  BitBoard theirs() const { return BitBoard(b_.occupied_black); }
  BitBoard pawns() const { return BitBoard(b_.pawns); }
  BitBoard en_passant() const { return BitBoard(0); }
  BitBoard bishops() const { return BitBoard(b_.bishops); }
  BitBoard rooks() const { return BitBoard(b_.rooks); }
  BitBoard queens() const { return BitBoard(b_.queens); }
  BitBoard knights() const { return BitBoard(b_.knights); }
  BitBoard kings() const { return BitBoard(b_.kings); }
  Castlings castlings() const { return Castlings(b_.castling_rights); }
  bool flipped() const { return !b_.turn_white; }  // Black to move

  // Direct access to the backing cc:: state (used by chess/board.cc & the backend).
  const cc::Board& cc() const { return b_; }
  cc::Board& cc() { return b_; }

  bool operator==(const ChessBoard& other) const;
  bool operator!=(const ChessBoard& other) const { return !(*this == other); }

 private:
  cc::Board b_;
};

std::string BoardToFen(const ChessBoard& board);

}  // namespace lczero
