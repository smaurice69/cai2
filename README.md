# Chiron Chess Engine

Chiron is a C++20 UCI chess engine designed for high-performance analysis, automated self-play, and in-engine evaluation training. The project bundles modern search features, a lightweight NNUE-style evaluator, data tooling, and extensive documentation to help you build, train, and analyse with a single binary.

## Features

* **UCI compatible** – Supports the complete UCI command set including ponder, time controls, hash/threads options, and asynchronous stop handling.
* **Advanced search** – Iterative deepening alpha-beta with aspiration windows, transposition table, quiescence search, killer/history move ordering, null-move pruning, late-move reductions, and configurable time management.
* **Self-play orchestration** – Runs many concurrent games with per-game logging (JSONL + PGN), resign/adjudication logic, and optional on-the-fly evaluator training.
* **Training pipeline** – Pure C++ NNUE-style trainer with dataset import/export, PGN conversion utilities, and an offline "teacher" bridge to external UCI engines such as Stockfish.
* **Extensive tooling** – Command-line entry points for perft validation, self-play, dataset generation, evaluator training, time-management analysis, and teacher annotation.
* **Cross-platform** – Builds with CMake on Linux, macOS, and Windows (MSVC) using only the standard library and GoogleTest for unit tests.

## Build Instructions

### Prerequisites

* A C++20 capable compiler (GCC 11+, Clang 13+, or MSVC 19.30+).
* CMake 3.20 or newer.
* Internet access during configuration to fetch GoogleTest.

### Configure & Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

#### Windows (Visual Studio 2022)

Open a **Developer Command Prompt for VS 2022** and generate the solution explicitly:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

You can open the resulting solution in Visual Studio or let the VS Code CMake Tools extension drive the build.

#### macOS (Apple Silicon)

Install the Xcode Command Line Tools (`xcode-select --install`) and ensure recent CMake binaries are available (e.g. `brew install cmake ninja`). Build a native arm64 binary with:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
ninja -C build
```

These presets work seamlessly inside VS Code's integrated terminal on macOS.

### Running Tests

```bash
cmake --build build --target chiron_tests
ctest --test-dir build
```

## UCI Usage

Launch the engine with no arguments to enter UCI mode:

```bash
./build/chiron
```

Supported UCI options:

* `Hash` (1–4096 MB)
* `Threads` (1–128)
* `Move Overhead` (ms)
* `Base Time Percent` (percentage of remaining time allocated per move)
* `Increment Percent` (percentage of increment invested per move)
* `Minimum Think Time` / `Maximum Think Time`
* `EvalNetwork` (path to NNUE network)
* `Ponder`

The engine honours `go` parameters for depth, movetime, ponder, and all time-control fields. Searches run asynchronously; `stop` or `ponderhit` commands interrupt the current search immediately.

## Command-Line Tools

The `chiron` executable also exposes a suite of helper commands:

| Command | Description |
|---------|-------------|
| `perft --depth N [--fen FEN]` | Executes a perft test from the current position. |
| `selfplay [options]` | Runs concurrent self-play games (see below). |
| `train --input dataset.txt [--output net.nnue] [--rate 0.05] [--batch 256] [--iterations 3] [--shuffle]` | Trains the evaluator on a dataset of `fen|score` lines. |
| `import-pgn --pgn games.pgn [--output dataset.txt] [--no-draws]` | Converts a PGN database into a training dataset. |
| `teacher --engine /path/to/uci --positions fens.txt [--output labels.txt] [--depth 20] [--threads 4]` | Calls an external UCI engine to annotate positions with evaluations. |
| `tune sprt ...` / `tune time ...` | Existing tuning utilities for SPRT matches and time-heuristic analysis. |

## Measuring Playing Strength

Use the SPRT harness to compare two binaries or network revisions and obtain an Elo estimate with confidence bounds:

```bash
./chiron tune sprt --games 400 --elo0 0 --elo1 20 \
  --baseline-name Baseline --candidate-name Candidate \
  --baseline-network nnue/models/baseline.nnue \
  --candidate-network nnue/models/chiron-selfplay-latest.nnue
```

The summary reports the SPRT conclusion, win/draw statistics, and the estimated Elo difference ± one 95% confidence interval, making it easy to track progress toward ambitious rating targets (3000+ Elo).

## Self-Play and Training

> Looking for a complete walkthrough? See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for an end-to-end recipe that covers data generation, training loops, and Elo evaluation.

Launch concurrent self-play with optional training:

```bash
./chiron selfplay \
  --games 200 \
  --depth 12 \
  --concurrency 4 \
  --enable-training \
  --training-batch 512 \
  --training-rate 0.05 \
  --training-output nnue/models/chiron-selfplay-latest.nnue \
  --training-history nnue/models/history \
  --results results.jsonl \
  --pgn games.pgn
```

Key options:

* `--concurrency N` – Number of worker threads playing games in parallel.
* `--threads N` / `--white-threads` / `--black-threads` – Search threads per engine.
* `--enable-training` – Collect FENs and periodically update the evaluator.
* `--training-batch SIZE` – Number of samples per optimisation step.
* `--training-rate RATE` – Learning rate for the internal trainer.
* `--training-output PATH` – Where to store the continually updated NNUE weights.
* `--training-history DIR` – Optional directory for archiving per-step snapshots.


* `--verbose` – Print per-move search telemetry and training updates during self-play.

Training batches are accumulated from every game (start position plus subsequent FENs). When the buffer exceeds the requested batch size, the trainer performs an optimisation step, saves the updated network, and reloads it for subsequent games.

### Verbose Telemetry

Enable `--verbose` to monitor self-play in real time. Each move is reported with its SAN notation, search depth/seldepth, evaluation (centipawns or mate score), node counts, NPS, elapsed time, and the current principal variation—useful insight for chess programmers inspecting move ordering and search stability. After every game, and whenever the training buffer flushes, verbose mode also prints how many positions were collected, the running totals processed, and where updated NNUE weights and history snapshots were written, giving AI practitioners clear feedback on dataset growth and model iteration cadence.

### Model Storage

By default the latest self-play network is written to `nnue/models/chiron-selfplay-latest.nnue`. Each optimisation step also writes a snapshot to `nnue/models/history/<prefix>-iterXXXX.nnue`. Both paths are configurable via `--training-output` and `--training-history`, and required directories are created automatically on Windows, macOS, and Linux.

## Dataset & Teacher Workflow

1. **Generate positions** – Run self-play with `--record-fens` or import existing PGNs with `import-pgn`.
2. **Label positions** – Optionally call a stronger engine with `teacher` to obtain supervised centipawn targets.
3. **Train** – Optimise NNUE weights using `train --input labels.txt --output new.nnue`.
4. **Deploy** – Point Chiron to the new network via `setoption name EvalNetwork value new.nnue` or supply it through `--training-output` in self-play.

## Developer Notes

* Public headers include Doxygen comments. Generate HTML docs with `cmake --build build --target doc` if Doxygen is available.
* The evaluation trainer, PGN importer, and teacher modules are pure C++ and require no Python tooling.
* All binaries accept `--help` style exploration by inspecting the README or source for supported options.

## Testing & Verification

Automated GoogleTest suites cover move generation (perft depth 1–6), self-play stability, and training save/load round-trips. Run them via `ctest` as shown above.

## License

This project is released under the MIT License. External engines invoked via the teacher tool remain independent executables; respect their licences accordingly.
