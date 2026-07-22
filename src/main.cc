/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "chess/board.h"
#include "default_search.h"
#include "engine.h"
#include "search/register.h"
#include "selfplay/loop.h"
#include "tools/benchmark.h"
// Chessckers: backendbench + onnx/proto tools (describenet/leela2onnx/onnx2leela)
// are excluded (proto/onnx-specific).
#include "utils/commandline.h"
#include "utils/esc_codes.h"
#include "utils/logging.h"
#include "utils/trace.h"
#include "version.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace lczero {
void ChooseAndRunEngine() {
  // First try the engine which is explicitly specified on the command line.
  for (const std::string_view search_name :
       SearchManager::Get()->GetSearchNames()) {
    if (CommandLine::ConsumeCommand(search_name)) {
      RunEngine(SearchManager::Get()->GetFactoryByName(search_name));
      return;
    }
  }

  // Then if DEFAULT_SEARCH is defined, run the engine specified by it.
#ifdef DEFAULT_SEARCH
  SearchFactory* factory =
      SearchManager::Get()->GetFactoryByName(DEFAULT_SEARCH);
  if (!factory) throw Exception("Unknown search algorithm: " DEFAULT_SEARCH);
  RunEngine(factory);
  return;
#endif

  // Then try to run the engine which is specified by the name of the binary.
  const std::string& binary_name = CommandLine::BinaryName();
  for (const std::string_view search_name :
       SearchManager::Get()->GetSearchNames()) {
    if (binary_name.find(search_name) != std::string::npos) {
      RunEngine(SearchManager::Get()->GetFactoryByName(search_name));
      return;
    }
  }

  // Finally, run "classic" search through the new API.
  RunEngine(SearchManager::Get()->GetFactoryByName("classic"));
}
}  // namespace lczero

int main(int argc, const char** argv) {
#ifdef __GLIBC__
  // Pin the malloc mmap threshold. The NN backend allocates a ~1MB Item buffer
  // per computation (AtomicVector of now-POD boards); glibc's DYNAMIC threshold
  // adapts upward on the first few frees, after which these buffers come from
  // the sbrk/arena heaps and the freed chunks are never returned to the OS —
  // ~165MB/s of retained RSS at CUDA eval rates (2026-07-22, post-POD-refactor
  // regression: the buffers crossed the threshold size when Board went POD).
  // Pinning disables the adaptation so >128KB blocks always mmap/munmap.
  mallopt(M_MMAP_THRESHOLD, 128 * 1024);
#endif
  LCTRACE_INITIALIZE;
  using namespace lczero;
  EscCodes::Init();
  LOGFILE << "akshay-chessckers-0 started.";
  CERR << EscCodes::Bold() << EscCodes::Red() << " _   _";
  CERR << "|_| |_|  akshay-chessckers-0" << EscCodes::Reset()
       << " v" << GetVersionStr()
       << " built " << __DATE__;

  try {
    // Chessckers: no magic bitboards (the cc:: core uses simple bitboard ops).

    CommandLine::Init(argc, argv);
    if (CommandLine::BinaryName().find("simple") == std::string::npos) {
      CommandLine::RegisterMode("selfplay", "Play games with itself");
      CommandLine::RegisterMode("benchmark", "Quick benchmark");
      CommandLine::RegisterMode("bench", "Very quick benchmark");
    }
    for (const std::string_view search_name :
         SearchManager::Get()->GetSearchNames()) {
      CommandLine::RegisterMode(
          std::string(search_name),
          "Use \"" + std::string(search_name) + "\" search");
    }

    if (CommandLine::ConsumeCommand("selfplay")) {
      // Selfplay mode.
      StdoutUciResponder uci_responder;
      SelfPlayLoop loop(&uci_responder);
      loop.Run();
    } else if (CommandLine::ConsumeCommand("benchmark")) {
      // Benchmark mode, longer version.
      Benchmark benchmark;
      benchmark.Run();
    } else if (CommandLine::ConsumeCommand("bench")) {
      // Benchmark mode, shorter version.
      Benchmark benchmark;
      benchmark.Run(/*run_shorter_benchmark=*/true);
    } else {
      lczero::ChooseAndRunEngine();
    }
  } catch (std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    abort();
  }
}
