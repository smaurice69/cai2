#pragma once

#include <memory>
#include <string>

#include "board.h"
#include "nnue/evaluator.h"

namespace chiron {

/**
 * @brief Evaluates the board using the configured NNUE network.
 *
 * Positive scores favour the side to move.
 */
int evaluate(const Board& board);

/**
 * @brief Returns a shared evaluator instance used by default throughout the engine.
 */
std::shared_ptr<nnue::Evaluator> global_evaluator();

/**
 * @brief Overrides the path of the NNUE network file used by the global evaluator.
 */
void set_global_network_path(const std::string& path);

}  // namespace chiron

