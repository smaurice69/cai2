#include <gtest/gtest.h>

#include "board.h"
#include "eval/evaluation.h"

TEST(EvaluationPipelineTest, StartPositionIsBalanced) {
    chiron::Board board;
    board.set_start_position();
    int eval = chiron::evaluate(board);
    EXPECT_NEAR(eval, 0, 50);
}

TEST(EvaluationPipelineTest, MaterialAdvantageReflectsScore) {
    chiron::Board board;
    board.set_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    int eval_white = chiron::evaluate(board);
    EXPECT_GT(eval_white, 400);

    board.set_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    int eval_black = chiron::evaluate(board);
    EXPECT_LT(eval_black, -400);
}
