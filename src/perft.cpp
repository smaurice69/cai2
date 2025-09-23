#include "perft.h"

namespace chiron {

std::uint64_t perft(Board& board, int depth) {
    if (depth == 0) {
        return 1ULL;
    }

    std::vector<Move> moves;
    MoveGenerator::generate_legal_moves(board, moves);

    std::uint64_t nodes = 0ULL;
    for (const Move& move : moves) {
        Board::State state;
        board.make_move(move, state);
        nodes += perft(board, depth - 1);
        board.undo_move(move, state);
    }
    return nodes;
}

}  // namespace chiron

