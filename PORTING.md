# Porting lc0 → akshay-chessckers-0

## STATUS (live)

**Builds, links, and PLAYS chessckers via lc0's classic MCTS + the chessckers
backend.** Done & verified: P0 rename; P1 board/move/position adapter (parity test);
P2 build-green; P3 NN backend (eval parity test + runtime); P4 search integration
(plays both the chess and checkers sides, opening double-move handled).
Remaining for full parity: **P5** self-play training-data output via `cc::chunk`
(currently the encoder is stubbed → `selfplay` runs but writes invalid chunks);
**P6** longer end-to-end parity vs the reference + perf (CPU nps is low; no GPU
batching yet through the lc0 backend).

### Build & run (macOS / Apple)
```
cd akshay-chessckers-0
meson setup build           # dag_classic defaults off now
meson configure build -Dblas=false -Dmetal=disabled -Dbuild_backends=false \
                      -Dgtest=false -Ddefault_backend=chessckers
ninja -C build
# play:
printf 'position startpos\ngo nodes 100\n' | \
  ./build/akshay-chessckers-0 --backend=chessckers --weights=<path/to/net.bin>
```
Standalone parity tests (no full build needed):
```
clang++ -std=c++20 -O2 -Isrc -Isrc/chessckers -DACCELERATE_NEW_LAPACK \
  tests_chessckers/<board_adapter|eval>_test.cc src/chess/board.cc src/chess/position.cc \
  -framework Accelerate -o /tmp/t && /tmp/t [<net.bin> for eval_test]
```

---


Converting the lc0 chess engine to play **Chessckers** (10×10-path checkers-vs-chess
hybrid; see `engine/chessckers.md` in the chessckers repo for the rules). This doc is
the authoritative design + checklist for the port so the "~10 things we can get wrong"
are written down, not rediscovered.

## Decisions (locked)

1. **NN representation: keep the reference engine's.** Variable per-position policy
   (one logit per legal move via a gather/feature move-encoder), 16-channel / 10×10
   position input. Existing trained nets (`engine/net-*.bin`) stay usable. We do **not**
   adopt lc0's fixed 1858-move conv-policy vector.
2. **Engine internals: force lc0's.** Keep lc0's `search/classic` scheduler, tree reuse,
   and the modern `Backend` abstraction; adapt them to the variable policy + a
   non-uint16 move type. lc0 contributes search + backends + UCI + engine loop + build +
   tooling; the chessckers reference (`src/chessckers/`, vendored from `engine/cpp`)
   contributes the game core (board/movegen/apply) + NN (encode/net) + selfplay/chunk.
3. **Everything that generalizes defers to lc0** (PUCT scheduling, threading, options,
   UCI). Chessckers-specific behavior defers to the reference engine.

## The crucial enabling discovery

lc0 0.33's backend seam is **already variable-policy**:

```cpp
struct EvalPosition { std::span<const Position> pos; std::span<const Move> legal_moves; };
struct EvalResult   { float q,d,m; std::vector<float> p; };   // p[i] aligned to legal_moves[i]
```

- `search.cc` builds `EvalPosition{history.GetPositions(), legal_moves}`, calls
  `AddInput(...)`, then writes `edge[i].SetP(eval->p[i])` **by index** — it never calls
  `MoveToNNIndex`. (`node.cc:185` `Edge::FromMovelist`; `search.cc:~1469,~2187`.)
- `MoveToNNIndex` / the 1858 `policy_map` / `encoder.cc` `kMoveStrs` are used **only**
  inside the *default* backend wrapper (`neural/wrapper.cc` `SoftmaxPolicy`) and
  `trainingdata`/`rescorer`. A **new backend that fills `p[i]` directly from the gather
  head bypasses all of it.**

So the search↔backend contract natively fits the chessckers gather head. The real work
is the *game types* and *one new backend*, not rewiring search's policy plumbing.

## Square indexing is 1:1 — no remap needed

`cc::Board` (reference `board.hpp`) uses `sq = rank*8 + file`, a1=0 … h8=63 — **identical
to lc0's `Square`**. Black checker towers ride the bitboards exactly as White chess pieces
(Stone-top = black pawn `p`, King-top = black king `k`). The **10×10 rim only exists for
Black capture *paths* (waypoints)**, which live *inside* a move (cadence/overshoot), never
as board squares. ⇒ lc0's 64-square `Square` and `BitBoard` (uint64) are reused as-is for
piece positions.

## Adapter surface (what the generic layers actually call)

From `src/search`, `src/selfplay`, `src/trainingdata`, `src/engine*`, `src/chess/uciloop`:

- **ChessBoard**: `SetFromFen`, `GenerateLegalMoves`, `GeneratePseudolegalMoves`,
  `ApplyMove(Move)->bool`, `IsUnderCheck`, `IsUnderAttack`, `GenerateKingAttackInfo`,
  `IsLegalMove`, `HasMatingMaterial`, `ParseMove(str)->Move`, `Hash`, `Mirror`,
  `ours/theirs/pawns/knights/bishops/rooks/queens/kings/en_passant`, `castlings`,
  `flipped`, `operator==`, `kStartposFen/kStartposBoard/kPawnMask`; free `BoardToFen`.
- **Move**: `from`, `to`, `ToString(bool)`, `Flip`, `is_null`, `operator==`,
  (+ `is_promotion/promotion/is_castling/is_en_passant` — encoder-only, droppable).
- **Position/PositionHistory**: `Position(parent,Move)`, `Position(board,r50,ply)`,
  `FromFen`, `Hash`, `IsBlackToMove`, `GetGamePly`, `GetRule50Ply`, `GetRepetitions`,
  `GetPliesSincePrevRepetition`, `GetBoard`; history `Last/Starting/GetPositions/Append/
  Pop/Reset/ComputeGameResult/HashLast/DidRepeatSinceLastZeroingMove`.

## Move representation

`lczero::Move` must carry a chessckers move (chains/deploys/charges/waypoints don't fit
uint16). Plan: wrap `std::shared_ptr<const cc::NativeMove>` (cheap copy into edges; heavy
struct shared). `ToString` = `nm->uci`; `from()/to()` read the structured src squares;
`operator==` compares uci; `Flip()` = **no-op**; `is_null()` = `!nm`. `ApplyMove` →
`cc::apply_native`. (Edge gets bigger — accepted per Decision 2.)

## The ~10 things we can get wrong (CHECKLIST)

1. **Asymmetry ⇒ no mirroring.** lc0 stores the board "from side-to-move" and `Mirror()`s
   after every move so "ours" is always to-move. Chessckers White=chess, Black=checkers —
   **there is no valid mirror.** Neuter `ChessBoard::Mirror()` and `Move::Flip()` to no-ops;
   keep the board in the absolute (White-frame) coordinates always. `flipped()` must mean
   "Black to move" *without* having mirrored anything. Re-audit every `Mirror`/`Flip` site
   (`node.cc:127,531-532`).
2. **No canonicalization / transforms.** Drop `encoder.cc` `ChooseTransform` & the 8-fold
   symmetry; chessckers has none. `transform` is always 0; `MoveToNNIndex(move,transform)`
   path is gone with the new backend.
3. **Variable policy length.** `EvalResult.p.size() == legal_moves.size()` per position;
   never a fixed 1858. Backend must size `p` per input.
4. **Extended FEN.** Stack overlay `[sq:pieces,…]` + trailing `{wm:N,r8:N}`. Use
   `cc::parse_fen/serialize_fen` verbatim; do **not** use lc0's FEN code.
5. **Opening double-move (`wm`).** White's first turn = two sub-moves; `apply_white_move`
   keeps the turn White while `white_moves_left>1`. `IsBlackToMove` must come from
   `turn_white`, not parity of ply.
6. **Rank-8 win counter (`r8`).** White wins by holding king on rank 8 for 3 of its turns
   w/o check; any check resets. Terminal detection lives in `apply.hpp` status, NOT lc0's
   `ComputeGameResult` chess rules.
7. **Terminal/result mapping.** `PositionHistory::ComputeGameResult` must map
   `cc::Status` (`variantEnd`/`mate`/`stalemate`/black-eliminated/king-captured) →
   lc0 `GameResult`. Don't reuse lc0's 50-move/insufficient-material/repetition draws
   blindly (`HasMatingMaterial` is chess-specific; chessckers terminals differ).
8. **Mandate (forced capture)** & **ram (suicide)**: emergent from `cc` movegen; just
   ensure `GenerateLegalMoves` calls `gen_legal_native` and nothing re-filters them with
   chess logic.
9. **Check predicate.** lc0's `IsUnderCheck`/`IsUnderAttack` are FIDE 8-direction; use
   `cc::white_in_chessckers_check` instead (Black attacks via checker hops/chains).
10. **Policy/value head wiring in the new backend.** Encode position via `cc::encode.hpp`
    V2 (16ch/10×10) + per-move features; run `cc::ChesskersNet`; return `p[i]` (logit per
    legal move, pre/post softmax per lc0's expectation — lc0 softmaxes in wrapper, so the
    new backend should output raw logits aligned to `legal_moves`), `q` (WDL→scalar), `d`,
    `m`. Net loads existing `.bin`.
11. **(bonus) Training-data format.** lc0 V6/V7 is fixed-1858 + chess planes. For parity
    with the reference, emit `cc::chunk.hpp` gzipped-JSON `ccz1` instead, or add a V-chunk
    writer. Rescorer (`trainingdata/rescorer.cc`) is chess-only → exclude from build.

## Phase plan (each phase ends green + verified)

- **P0 ✅** rename product identity → `akshay-chessckers-0`.
- **P1 ✅(1a)** vendor `engine/cpp` core into `src/chessckers/` (compiles standalone).
  **(1b)** game-type adapter: `types.h` Move, `board.{h,cc}`, `position.cc` over `cc::`,
  neutering Mirror/Flip; standalone parity test vs direct `cc::` (movegen UCIs + applied
  FENs + terminal) on a FEN corpus.
- **P2** wire movegen/apply/status fully; exclude chess-only TUs (syzygy, rescorer,
  lc0 `board.cc` magic-bitboard movegen, encoder/decoder/policy_map) from meson.
- **P3** new `Backend` (`neural/backends/chessckers/`) wrapping `cc::ChesskersNet` +
  `cc::encode`; loads `.bin`; returns `EvalResult{q,d,m,p[]}`. Register via factory.
- **P4** make `search/classic` compile+run on the new types (Edge move storage, drop
  encoder/transform calls); reconcile PUCT constants (cc: cpuct 2.0/base 19652/fpu 0.25).
- **P5** selfplay + training data → `cc::chunk` parity.
- **P6** full meson build green; byte-parity vs `engine/cpp` + PyVariant oracle on shared
  FENs (movegen, encode, a short self-play game).

## Build notes

- Core is header-only; deps: BLAS (Accelerate on mac / OpenBLAS else), zlib, threads,
  optional Metal (`nn_metal.mm`). Add include dir `src/chessckers` + these deps to the
  meson targets that pull the new backend/selfplay TUs.
- Excluded-from-build (chess-only, replaced): `chess/board.cc` magic movegen internals,
  `neural/encoder.cc`/`decoder.cc`/`tables/policy_map.h`, `syzygy/`, `trainingdata/rescorer*`.
</content>
