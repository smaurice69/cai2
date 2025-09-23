#include "evaluation.h"

namespace chiron {

namespace {
constexpr int kPieceValues[static_cast<int>(PieceType::King) + 1] = {100, 320, 330, 500, 900, 20000};
}

int evaluate(const Board& board) {
    int score = 0;
    for (int piece = 0; piece < kNumPieceTypes; ++piece) {
        PieceType type = static_cast<PieceType>(piece);
        int value = kPieceValues[piece];
        score += value * popcount(board.pieces(Color::White, type));
        score -= value * popcount(board.pieces(Color::Black, type));
    }

    return board.side_to_move() == Color::White ? score : -score;
}

}  // namespace chiron

