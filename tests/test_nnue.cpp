#include <gtest/gtest.h>

#include "board.h"
#include "evaluation.h"

namespace chiron {

TEST(NnueEvaluation, StartPositionIsBalanced) {
    Board board;
    board.set_start_position();
    EXPECT_EQ(evaluate(board), 0);
}

TEST(NnueEvaluation, MaterialAdvantageReflectsScore) {
    Board board;
    board.set_from_fen("8/8/8/8/8/8/4P3/7K w - - 0 1");
    EXPECT_GT(evaluate(board), 0);

    board.set_from_fen("8/8/8/8/8/8/4p3/7k w - - 0 1");
    EXPECT_LT(evaluate(board), 0);
}

}  // namespace chiron

