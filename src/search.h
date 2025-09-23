#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "board.h"
#include "movegen.h"
#include "nnue/evaluator.h"

namespace chiron {

/**
 * @brief Minimal negamax alpha-beta search driver with an internal transposition table.
 */
class Search {
   public:
    explicit Search(std::size_t table_size = 1 << 20, std::shared_ptr<nnue::Evaluator> evaluator = nullptr);

    Move search_best_move(Board& board, int max_depth);
    void clear();

    void set_evaluator(std::shared_ptr<nnue::Evaluator> evaluator) { evaluator_ = std::move(evaluator); }

   private:
    struct TTEntry {
        std::uint64_t key = 0ULL;
        int depth = -1;
        int score = 0;
        Move move{};
        enum class Flag : std::uint8_t { Empty, Exact, Alpha, Beta } flag = Flag::Empty;
    };

    std::vector<TTEntry> table_;

    std::shared_ptr<nnue::Evaluator> evaluator_;
    std::vector<nnue::Accumulator> accumulator_stack_;

    [[nodiscard]] TTEntry& entry_for_key(std::uint64_t key);
    [[nodiscard]] const TTEntry& entry_for_key(std::uint64_t key) const;

    int alpha_beta(Board& board, int depth, int alpha, int beta, int ply);
    void store_tt(std::uint64_t key, int depth, int score, const Move& move, TTEntry::Flag flag);
    bool probe_tt(std::uint64_t key, TTEntry& out) const;

    Move best_move_{};
};

}  // namespace chiron

