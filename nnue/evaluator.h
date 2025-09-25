#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "board.h"
#include "move.h"
#include "nnue/network.h"

namespace chiron::nnue {

/**
 * @brief Accumulator storing the summed NNUE feature contributions for both colors.
 */
struct Accumulator {
    std::array<int32_t, kNumColors> material{};

    void reset() { material.fill(0); }
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

