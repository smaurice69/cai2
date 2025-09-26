# Training Chiron to Play Strong Chess

This guide walks you through building a complete self-play and supervised-training loop for the Chiron chess engine. The focus is on repeatable workflows that run on macOS (Apple Silicon) and Windows (Visual Studio 2022 or VS Code + MSVC), steadily improving the bundled NNUE network until the engine reaches competitive strength.

## 1. Prepare the Environment

1. **Build the engine** using the steps in the main `README.md` (CMake + Ninja/Visual Studio). On Windows, build from the "Developer Command Prompt for VS 2022"; on macOS use the arm64 toolchain (`cmake -G Ninja -DCMAKE_OSX_ARCHITECTURES=arm64`).
2. **Create output directories** if they are not present:
   ```bash
   mkdir -p nnue/models/history logs data
   ```
   These folders are Git-ignored so your iterative networks, datasets, and logs will not pollute the repository.
3. (Optional) **Install a strong teacher engine** such as the latest Stockfish build if you plan to label positions for supervised training. Detailed setup instructions are provided in [Section 3.1](#31-integrate-stockfish-as-a-teacher).

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
     --verboselite
   ```
2. The orchestrator automatically streams search telemetry (with `--verbose`), saves the continually updated network to `nnue/models/chiron-selfplay-latest.nnue`, and writes snapshot checkpoints to `nnue/models/history/` every optimisation step. This ensures each training run builds on the previous weights instead of starting from scratch. Swap `--verboselite` for `--verbose` if you only need to know when games finish without the detailed move trace.

## 3. Grow the Training Dataset

1. **Self-play expansion** – Increase `--games`, `--depth`, and `--concurrency` as your hardware allows, and add `--record-fens data/selfplay_fens.txt` to accumulate labelled positions. More games diversify the dataset and stabilise learning.
2. **External data** – Convert public PGN dumps or lichess puzzles into training samples. See [Section 3.2](#32-build-a-training-set-from-public-chess-databases) for a full workflow covering sourcing, filtering, and conversion.
3. **Teacher annotations** – Use a stronger engine to annotate FENs gathered from self-play for supervised learning. Refer to [Section 3.1](#31-integrate-stockfish-as-a-teacher) for a step-by-step example with Stockfish:
   ```bash
   ./build/chiron teacher \
     --engine /path/to/stockfish \
     --positions data/selfplay_fens.txt \
     --output data/labels.txt \
     --depth 20 \
     --threads 4
   ```

### 3.1 Integrate Stockfish as a Teacher

Stockfish offers world-class evaluation quality and can dramatically improve Chiron's supervised dataset. The typical workflow is:

1. **Download Stockfish**
   * **Linux/macOS:** download the latest prebuilt binary from <https://stockfishchess.org/download/>.
     ```bash
     curl -L -o stockfish.tar.zst "https://stockfishchess.org/files/stockfish-ubuntu-x86-64-avx2.tar.zst"
     tar --zstd -xf stockfish.tar.zst
     mv stockfish-ubuntu-x86-64-avx2/bin/stockfish tools/stockfish
     ```
   * **Windows:** fetch `stockfish-windows-x86-64.zip`, extract `stockfish.exe`, and place it beside the Chiron binary (for example `build/stockfish.exe`).

   To build from source instead, clone the Stockfish repository and run `cmake -S src -B build` followed by `cmake --build build --target build` (or simply `make build` on Linux). Ensure the resulting executable is on your `PATH` or record its absolute path for later.

2. **Verify UCI connectivity**
   ```bash
   ./tools/stockfish uci
   ```
   Stockfish should print `uciok` within a second; exit with `quit`. If you see a permissions error, run `chmod +x tools/stockfish` on Linux/macOS or unblock the file in Windows Defender SmartScreen.

3. **Prepare positions for annotation**
   Generate and deduplicate FENs before sending them to the teacher to avoid wasted effort:
   ```bash
   ./build/chiron selfplay --games 100 --record-fens data/selfplay_raw.fens
   awk '!seen[$0]++' data/selfplay_raw.fens > data/selfplay_unique.fens
   ```

4. **Annotate with the teacher tool**
   ```bash
   ./build/chiron teacher \
     --engine ./tools/stockfish \
     --positions data/selfplay_unique.fens \
     --output data/stockfish_labels.txt \
     --depth 25 \
     --threads 8 \
     --nodes 2000000
   ```
   * Increase `--threads` to match available CPU cores; Chiron spawns one Stockfish instance per worker.
   * Use `--depth` or `--nodes` to control runtime. Node limits yield consistent throughput on mixed hardware.
   * Add `--multipv 2` (or higher) if you want the teacher to report multiple candidate moves per position.

5. **Record teacher metadata**
   Save the Stockfish version and benchmark inside a sidecar file so you can reproduce results later:
   ```bash
   ./tools/stockfish bench | tee data/stockfish_labels.txt.metadata
   ```

### 3.2 Build a Training Set from Public Chess Databases

Curated game collections quickly expand your supervised dataset. The pipeline below works on Linux, macOS, and Windows (PowerShell):

1. **Select a source**
   * [Lichess Elite Database](https://database.lichess.org/) – monthly PGNs from strong online players.
   * [KingBase Lite](https://www.kingbase-chess.net/) – classical over-the-board games by titled players.
   * [FICS/ICC archives](https://www.ficsgames.org/download.html) – large blitz/rapid sets for tactical diversity.

2. **Download and verify**
   ```bash
   mkdir -p data/databases
   curl -L -o data/databases/lichess_elite.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
   curl -L -o data/databases/lichess_elite.sha256 https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.sha256
   sha256sum -c data/databases/lichess_elite.sha256
   ```
   Expect `OK` on the checksum line. On Windows, use `CertUtil -hashfile lichess_elite.pgn.zst SHA256` instead of `sha256sum`.

3. **Decompress and filter**
   ```bash
   zstd -d data/databases/lichess_elite.pgn.zst -o data/databases/lichess_elite.pgn
   ./build/chiron import-pgn \
     --pgn data/databases/lichess_elite.pgn \
     --output data/lichess_elite.txt \
     --min-elo 2200 \
     --result-filter decisive \
     --max-games 500000
   ```
   * Lower `--min-elo` to widen the pool; raise it for higher-quality expert play.
   * `--result-filter decisive` discards draws—omit it for balanced datasets.
   * Add `--no-metadata` to skip comment parsing when speed matters.

4. **Merge and deduplicate**
   ```bash
   cat data/lichess_elite.txt data/kingbase.txt > data/external_raw.txt
   ./tools/dataset_dedupe.py --input data/external_raw.txt --output data/external_unique.txt
   ```
   `dataset_dedupe.py` can be a simple Python utility that tracks seen FENs; keep it in `tools/` for reuse.

5. **Blend with self-play and teacher data**
   ```bash
   cat data/external_unique.txt data/stockfish_labels.txt data/selfplay_labels.txt > data/combined_dataset.txt
   shuf data/combined_dataset.txt > data/combined_shuffled.txt
   ```
   Shuffling prevents long runs of similar positions and improves generalisation during training.

6. **Document provenance**
   Maintain a short README in `data/databases/` noting download URLs, checksums, filters, and extraction dates. This makes future experiments reproducible and simplifies sharing datasets with collaborators.

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
3. Replace the runtime network by pointing self-play or the UCI option to the freshly trained weights. Pass `--training-hidden <SIZE>` during self-play (or `--hidden <SIZE>` to the offline `train` command) to start from a wider NNUE with more hidden neurons when you want additional model capacity.
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
