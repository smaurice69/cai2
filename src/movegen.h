#pragma once

#include <vector>

#include "board.h"

namespace chiron {

/**
 * @brief Generates legal chess moves for the given board state.
 */
class MoveGenerator {
   public:
    static void generate_legal_moves(Board& board, std::vector<Move>& moves);
    static std::vector<Move> generate_legal_moves(Board& board);

   private:
    static void generate_pseudo_legal_moves(const Board& board, std::vector<Move>& moves);
    static void add_promotion_moves(int from, int to, bool is_capture, std::vector<Move>& moves);
};

}  // namespace chiron

