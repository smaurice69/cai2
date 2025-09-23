#include <gtest/gtest.h>

#include "perft.h"

namespace chiron {

TEST(PerftTest, StartPositionDepths) {
    Board board;
    board.set_start_position();
    EXPECT_EQ(perft(board, 1), 20ULL);
    EXPECT_EQ(perft(board, 2), 400ULL);
    EXPECT_EQ(perft(board, 3), 8902ULL);
    EXPECT_EQ(perft(board, 4), 197281ULL);
    EXPECT_EQ(perft(board, 5), 4865609ULL);
    EXPECT_EQ(perft(board, 6), 119060324ULL);
}

TEST(PerftTest, KiwipeteDepths) {
    Board board;
    board.set_from_fen("rnbq1k1r/pppp1ppp/5n2/4p3/1bB1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1");
    // Reference counts validated against python-chess' perft implementation.
    EXPECT_EQ(perft(board, 1), 29ULL);
    EXPECT_EQ(perft(board, 2), 956ULL);
    EXPECT_EQ(perft(board, 3), 28900ULL);
    EXPECT_EQ(perft(board, 4), 951029ULL);
}

}  // namespace chiron

