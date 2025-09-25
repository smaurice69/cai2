#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "board.h"
#include "move.h"
#include "nnue/network.h"

namespace chiron::nnue {

// Clamp evaluations so runaway training updates cannot be mistaken for mate scores.
constexpr int kMaxEvaluationMagnitude = 30000;

/**
 * @brief Accumulator storing the summed NNUE feature contributions for both colors.
 */
struct Accumulator {
    std::vector<int32_t> white;
    std::vector<int32_t> black;

    void reset(std::size_t hidden_size) {
        white.assign(hidden_size, 0);
        black.assign(hidden_size, 0);
    }
};

/**
 * @brief High-level evaluator that wraps a lightweight NNUE network.
 */
class Evaluator {
   public:
    Evaluator();

    void set_network_path(std::string path);
    void ensure_network_loaded() const;

    void build_accumulator(const Board& board, Accumulator& accum) const;
    void update_accumulator(const Board& board, const Move& move, const Accumulator& base, Accumulator& dest) const;

    int evaluate(const Board& board, const Accumulator& accum) const;

    [[nodiscard]] const Network& network() const;

   private:
    void apply_feature(Accumulator& accum, Color color, PieceType piece, int square, int sign) const;

    std::string network_path_;
    mutable Network network_{};
    mutable std::atomic<bool> network_loaded_{false};
    mutable std::mutex load_mutex_;
};

}  // namespace chiron::nnue

