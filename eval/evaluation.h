#pragma once

#include "board.h"

namespace chiron {

/**
 * @brief Evaluates the board using a simple material balance heuristic.
 * @return Positive scores favor the side to move.
 */
int evaluate(const Board& board);

}  // namespace chiron

