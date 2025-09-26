#include <algorithm>

#include <gtest/gtest.h>

#include "board.h"
#include "movegen.h"
#include "search.h"

namespace chiron {

namespace {

bool contains_move(const std::vector<Move>& moves, const Move& target) {
    return std::any_of(moves.begin(), moves.end(), [&](const Move& move) {
        return move.from == target.from && move.to == target.to && move.promotion == target.promotion &&
               move.flags == target.flags;
    });
}

}  // namespace

TEST(SearchIntegration, ProducesLegalMoveFromStartPosition) {
    Board board;
    board.set_start_position();

    Search search;
    SearchLimits limits;
    limits.max_depth = 2;

    SearchResult result = search.search(board, limits);

    std::vector<Move> legal = MoveGenerator::generate_legal_moves(board);
    EXPECT_FALSE(legal.empty());
    EXPECT_TRUE(contains_move(legal, result.best_move));
}

TEST(SearchNullMove, PreservesAccumulatorState) {
    Board board;
    board.set_from_fen("8/8/8/8/8/8/PPP5/K6k w - - 0 1");

    Search search;
    constexpr int depth = 3;
    constexpr int alpha = 0;
    constexpr int beta = 50;

    int score = SearchTestHelper::negamax_entry(search, board, depth, alpha, beta);
    EXPECT_GE(score, beta) << "Null-move pruning failed to trigger with consistent evaluation.";
}

}  // namespace chiron

