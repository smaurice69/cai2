# Training Run Analysis

This document summarizes the observations from the ten-iteration `learn` run (depth 10, 32 workers) provided in the latest report and lists follow-up work.

## Observations

- **Self-play results repeat every iteration.** Each block of eight games reports identical Elo deltas and a clean sweep for the side starting as White. The tracker resets to ~1512 vs ~1488 at the start of every iteration because we spin up a fresh `SelfPlayOrchestrator`, so the apparent Elo gain is not cumulative. However, the more important point is that the outcome pattern does not change at all between iterations, which means self-play is not generating new experience. 【F:training/TRAINING_ANALYSIS.md†L1-L8】
- **Hold-out and online replay metrics are flat.** The pseudo-Elo stays at 798.3 with the printed accuracy rounded to 100.0% on every pass. This shows the network is already confident on the hold-out slice and the online batches are not shifting the prediction distribution. 【F:training/TRAINING_ANALYSIS.md†L10-L12】
- **Cumulative supervised count only advances during online replay.** After each iteration the "Cumulative supervised samples" total increases by exactly 2048 (the PGN replay batch size). Self-play batches are either not large enough to trigger intermediate training updates or their contributions are not being surfaced in this counter, so we have no visibility into whether self-play examples are being used effectively. 【F:training/TRAINING_ANALYSIS.md†L14-L16】

## Suggested follow-up tasks

1. **Add instrumentation around self-play training.** Log how many positions are collected, when `train_buffer_if_ready_locked(true)` fires at the end of an iteration, and how large the flush was. This will confirm whether the self-play samples are actually hitting the trainer or if they are dropped because the buffer never reaches the configured batch size. 【F:training/TRAINING_ANALYSIS.md†L18-L21】
2. **Persist and compare Elo history across iterations.** Either reuse the `EloTracker` instance or aggregate the self-play PGNs so we can see whether the new network outperforms the previous one instead of re-estimating Elo from scratch every time. That will expose real progress (or the lack thereof) from one iteration to the next. 【F:training/TRAINING_ANALYSIS.md†L22-L24】
3. **Audit the online/hold-out datasets.** Sample the generated FENs and verify we get a healthy mix of side-to-move labels and evaluation targets instead of 99% win/loss leaf positions. If the importer still produces monotonous targets, train batches will converge instantly and stop providing gradient signal, which matches the flat pseudo-Elo curve. 【F:training/TRAINING_ANALYSIS.md†L25-L28】
4. **Introduce diversity into self-play.** Enable move randomness (temperature, top-N, score margin) during the opening phase and consider swapping colors on every game pair to prevent deterministic repeats where White always wins. More diverse self-play trajectories should yield novel positions and training signal. 【F:training/TRAINING_ANALYSIS.md†L29-L31】
5. **Track a richer metric (e.g., MSE) for the trainer.** Pseudo-Elo rounded to one decimal hides small movements. Recording mean squared error or cross-entropy on both the self-play and PGN batches would give us a sensitive indicator of progress. 【F:training/TRAINING_ANALYSIS.md†L32-L33】
