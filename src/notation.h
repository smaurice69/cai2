#pragma once

#include <string>

#include "board.h"

namespace chiron {

/**
 * @brief Converts a move to Standard Algebraic Notation (SAN).
 *
 * The board state is updated temporarily to determine check/checkmate markers and
 * then restored before returning.
 *
 * @param board Board instance on which the move will be played.
 * @param move  Legal move to convert.
 * @return SAN string representation of the move.
 */
std::string move_to_san(Board& board, const Move& move);

/**
 * @brief Parses a SAN string into a concrete legal move for the current board position.
 *
 * @param board Board containing the position to parse within.
 * @param san   SAN string describing a legal move.
 * @return Parsed move matching the SAN description.
 * @throws std::runtime_error if no legal move matches the SAN text.
 */
Move san_to_move(Board& board, const std::string& san);

}  // namespace chiron

