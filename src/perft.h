#pragma once

#include <cstdint>

#include "board.h"
#include "movegen.h"

namespace chiron {

/**
 * @brief Runs a perft (performance test) count to validate move generation.
 */
std::uint64_t perft(Board& board, int depth);

}  // namespace chiron

