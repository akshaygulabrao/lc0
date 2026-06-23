# Building the Chessckers fork (akshay-chessckers-0)

The fork is an lc0-derivative C++ engine. It self-plays on GPU boxes, produces
ccz1 game chunks, and evaluates positions via `cc::ChesskersNet`. It is **not**
the trainer — the trainer lives in `engine/chessckers_engine/train_continuous.py`.

## Prerequisites

```bash
# Build deps (Ubuntu)
apt-get install -y build-essential libopenblas-dev

# meson + ninja (pip, not apt — apt's meson is too old)
pip3 install meson ninja

# Go (for the lczero-client, separate repo; the fork itself is C++ only)
# See lczero-client/README.md
```

## Build

The fork uses meson. **Disable the legacy lc0 backends** — they reference symbols
(`NetworkAsBackendFactory`, `MultiHeadWeights`) that were removed when the
chessckers backend replaced lc0's fixed-policy NN stack.

```bash
meson setup build/release --buildtype release -Dblas=false -Dplain_cuda=false -Donnx=false -Dbuild_backends=false
ninja -C build/release akshay-chessckers-0
```

The binary is `build/release/akshay-chessckers-0`.

### One-liner (the `build.sh` wrapper)

```bash
bash build.sh
```

…but `build.sh` passes no flags → it'll try to build the legacy backends and
fail. Either pass the flags yourself or reconfigure after the fact:

```bash
bash build.sh
meson configure build/release -Dblas=false -Dplain_cuda=false -Donnx=false -Dbuild_backends=false
ninja -C build/release akshay-chessckers-0
```

## Starting position

The training start FEN lives in `src/chess/board.cc`:

```cpp
const char* ChessBoard::kStartposFen =
    "3kk3/8/8/8/8/8/8/4K3[d8:kk,e8:kk] w - - 0 1";
```

Change it, rebuild (`ninja -C build/release akshay-chessckers-0`), and restart
the self-play client(s). The trainer and server do **not** need a rebuild — only
the engine that plays the games.

## How the client finds this engine

The Go client (`lczero-client`) discovers the engine via a symlink:

```
lczero-client/.enginebin/akshay-chessckers-0 -> build/release/akshay-chessckers-0
```

The client adds `.enginebin/` to `PATH` before spawning the engine. No `--engine`
flag exists. See `lczero-client/README.md`.

## Architecture notes

- **Move generation / rules**: `src/chessckers/movegen.hpp`, `board.hpp`
- **NN encoding**: `src/chessckers/encode.hpp` (channels 8–12 = per-depth tower stack)
- **NN forward (CPU BLAS)**: `src/chessckers/nn.hpp` (loads `.bin` weights)
- **NN forward (GPU)**: `src/chessckers/nn_cuda.{hpp,cu}` (CUDA trunk + CPU heads)
- **Eval glue for lc0 search**: `src/chessckers/lc0_eval.hpp`
- **lc0 backend adapter**: `src/neural/backends/chessckers_backend.cc`
- **Encoder stub (unused, link only)**: `src/neural/encoder_chessckers_stub.cc`

The `meson.build` excludes lc0's fixed-policy encoder/decoder and the
`NetworkAsBackend` wrapper — the chessckers backend is self-contained.
`wrapper.cc` is kept (it defines `NetworkAsBackendFactory`, needed by the
`REGISTER_NETWORK` macro even though those backends aren't built), and the
legacy backends are disabled via meson flags.
