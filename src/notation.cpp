#include "notation.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "movegen.h"

namespace chiron {

namespace {

char piece_to_char(PieceType piece) {
    switch (piece) {
        case PieceType::Knight:
            return 'N';
        case PieceType::Bishop:
            return 'B';
        case PieceType::Rook:
            return 'R';
        case PieceType::Queen:
            return 'Q';
        case PieceType::King:
            return 'K';
        case PieceType::Pawn:
        case PieceType::None:
        default:
            return '\0';
    }
}

std::string canonicalize(const std::string& san) {
    std::string trimmed;
    trimmed.reserve(san.size());
    for (char c : san) {
        if (c == '+' || c == '#') {
            continue;
        }
        if (c == '!' || c == '?') {
            continue;
        }
        trimmed += c;
    }
    return trimmed;
}

std::string format_move(Board& board, const Move& move) {
    if (move.is_castle()) {
        return (move.flags & MoveFlag::KingCastle) ? "O-O" : "O-O-O";
    }

    PieceType moving_piece = board.piece_type_at(move.from);
    std::string san;
    if (moving_piece != PieceType::Pawn) {
        san += piece_to_char(moving_piece);

        auto legal_moves = MoveGenerator::generate_legal_moves(board);
        bool needs_file = false;
        bool needs_rank = false;
        bool conflict = false;
        for (const Move& candidate : legal_moves) {
            if (candidate.to == move.to && candidate.from != move.from) {
                PieceType candidate_piece = board.piece_type_at(candidate.from);
                if (candidate_piece == moving_piece) {
                    conflict = true;
                    if ((candidate.from & 7) == (move.from & 7)) {
                        needs_file = true;
                    }
                    if ((candidate.from >> 3) == (move.from >> 3)) {
                        needs_rank = true;
                    }
                }
            }
        }
        if (conflict) {
            if (!needs_file) {
                san += static_cast<char>('a' + (move.from & 7));
            } else if (!needs_rank) {
                san += static_cast<char>('1' + (move.from >> 3));
            } else {
                san += static_cast<char>('a' + (move.from & 7));
                san += static_cast<char>('1' + (move.from >> 3));
            }
        }
    } else if (move.is_capture()) {
        san += static_cast<char>('a' + (move.from & 7));
    }

    if (move.is_capture()) {
        san += 'x';
    }
    san += square_to_string(static_cast<Square>(move.to));

    if (move.is_promotion()) {
        san += '=';
        san += piece_to_char(move.promotion);
    }

    Board::State state;
    board.make_move(move, state);
    bool opponent_in_check = board.in_check(board.side_to_move());
    bool opponent_has_moves = !MoveGenerator::generate_legal_moves(board).empty();
    board.undo_move(move, state);

    if (opponent_in_check) {
        san += opponent_has_moves ? '+' : '#';
    }

    return san;
}

}  // namespace

std::string move_to_san(Board& board, const Move& move) { return format_move(board, move); }

Move san_to_move(Board& board, const std::string& san) {
    const std::string canonical = canonicalize(san);
    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board);
    for (const Move& move : moves) {
        std::string candidate = canonicalize(move_to_san(board, move));
        if (candidate == canonical) {
            return move;
        }
    }
    throw std::runtime_error("No legal move matches SAN: " + san);
}

}  // namespace chiron

