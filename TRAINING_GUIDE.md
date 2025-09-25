# Training Chiron to Play Strong Chess

This guide walks you through building a complete self-play and supervised-training loop for the Chiron chess engine. The focus is on repeatable workflows that run on macOS (Apple Silicon) and Windows (Visual Studio 2022 or VS Code + MSVC), steadily improving the bundled NNUE network until the engine reaches competitive strength.

## 1. Prepare the Environment

1. **Build the engine** using the steps in the main `README.md` (CMake + Ninja/Visual Studio). On Windows, build from the "Developer Command Prompt for VS 2022"; on macOS use the arm64 toolchain (`cmake -G Ninja -DCMAKE_OSX_ARCHITECTURES=arm64`).
2. **Create output directories** if they are not present:
   ```bash
   mkdir -p nnue/models/history logs data
   ```
   These folders are Git-ignored so your iterative networks, datasets, and logs will not pollute the repository.
3. (Optional) **Install a strong teacher engine** such as the latest Stockfish build if you plan to label positions for supervised training.

## 2. Kick-Start the Network

Chiron ships with a lightweight NNUE evaluator. To bootstrap training:

1. Run a short burst of self-play to collect positions and produce an initial network:
   ```bash
   ./build/chiron selfplay \
     --games 50 \
     --depth 8 \
     --concurrency 2 \
     --enable-training \
     --training-batch 256 \
     --training-rate 0.05 \
     --training-output nnue/models/chiron-selfplay-latest.nnue \
     --training-history nnue/models/history \
     --results logs/results.jsonl \
     --pgn logs/selfplay.pgn \
     --verbose
   ```
2. The orchestrator automatically streams search telemetry (with `--verbose`), saves the continually updated network to `nnue/models/chiron-selfplay-latest.nnue`, and writes snapshot checkpoints to `nnue/models/history/` every optimisation step. This ensures each training run builds on the previous weights instead of starting from scratch.

## 3. Grow the Training Dataset

1. **Self-play expansion** – Increase `--games`, `--depth`, and `--concurrency` as your hardware allows, and add `--record-fens data/selfplay_fens.txt` to accumulate labelled positions. More games diversify the dataset and stabilise learning.
2. **External data** – Convert public PGN dumps or lichess puzzles into training samples:
   ```bash
   ./build/chiron import-pgn --pgn master_games.pgn --output data/master.txt
   ```
3. **Teacher annotations** – Use a stronger engine to annotate FENs gathered from self-play for supervised learning:
   ```bash
   ./build/chiron teacher \
     --engine /path/to/stockfish \
     --positions data/selfplay_fens.txt \
     --output data/labels.txt \
     --depth 20 \
     --threads 4
   ```

## 4. Train the Evaluator Offline

1. Aggregate datasets (self-play, imported, teacher-labelled) into a single text file of `fen|score` lines.
2. Launch the trainer:
   ```bash
   ./build/chiron train \
     --input data/combined.txt \
     --output nnue/models/offline-latest.nnue \
     --batch 1024 \
     --iterations 8 \
     --rate 0.03 \
     --shuffle
   ```
3. Replace the runtime network by pointing self-play or the UCI option to the freshly trained weights:
   ```bash
   ./build/chiron selfplay --enable-training --training-output nnue/models/offline-latest.nnue
   ```
   or
   ```
   setoption name EvalNetwork value nnue/models/offline-latest.nnue
   ```

## 5. Continuous Improvement Loop

1. **Iterate self-play** – Keep `--enable-training` on so each batch of games refines `chiron-selfplay-latest.nnue`. Preserve history snapshots for rollback and regression testing.
2. **Mix data sources** – Periodically merge human/teacher-labelled datasets with fresh self-play positions to prevent overfitting to engine biases.
3. **Tune hyperparameters** – Experiment with batch sizes (256–2048), learning rates (0.01–0.10), and search depths to balance data quality against throughput. Apple Silicon and modern Windows desktops can comfortably run 4–8 concurrent games while training in the background.

## 6. Measure Progress Toward 3000+ Elo

1. Use the built-in SPRT harness to evaluate new networks against a trusted baseline:
   ```bash
   ./build/chiron tune sprt \
     --games 400 \
     --elo0 0 --elo1 20 \
     --baseline-name Baseline --candidate-name Candidate \
     --baseline-network nnue/models/baseline.nnue \
     --candidate-network nnue/models/chiron-selfplay-latest.nnue
   ```
2. Track the reported Elo delta and 95% confidence interval. Promote new networks only when the test conclusively favours the candidate.
3. Optionally log key metrics (win rate, plies searched, node throughput) from verbose self-play to identify stagnation early.

## 7. Operational Tips

* **Cross-platform parity** – The same binaries and command lines work on macOS (arm64) and Windows (x64) thanks to the engine's standard-library dependencies. Use VS Code's CMake Tools to keep build configurations aligned across machines.
* **Checkpoint hygiene** – Periodically prune old entries in `nnue/models/history` while archiving milestone networks externally.
* **Automation** – Script your self-play + training runs (PowerShell on Windows, shell scripts on macOS/Linux) so nightly builds collect games, retrain, and test automatically.
* **Data quality first** – Strong networks need diverse, high-quality positions. Mix tactical, strategic, and endgame scenarios; enforce adjudication rules (e.g., resign when eval < -8 for 6 plies) to avoid noisy blunders.

Following this cycle—generate data, train, evaluate, and iterate—will steadily raise Chiron's playing strength, setting the stage for a >3000 Elo engine once compute resources and training time scale accordingly.
