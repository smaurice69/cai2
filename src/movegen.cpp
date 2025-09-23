#include "movegen.h"

#include <algorithm>

#include "attacks.h"

namespace chiron {

void MoveGenerator::generate_pseudo_legal_moves(const Board& board, std::vector<Move>& moves) {
    moves.clear();

    Color us = board.side_to_move();
    Color them = opposite_color(us);
    Bitboard friendly = board.occupancy(us);
    Bitboard enemy = board.occupancy(them);
    Bitboard occupied = board.occupancy_all();

    // Pawn moves
    Bitboard pawns = board.pieces(us, PieceType::Pawn);
    while (pawns) {
        int from = pop_lsb(pawns);
        int rank = from / 8;
        int forward = from + (us == Color::White ? 8 : -8);
        if (forward >= 0 && forward < kBoardSize && board.piece_type_at(forward) == PieceType::None) {
            bool promotion_rank = (us == Color::White) ? (rank == 6) : (rank == 1);
            if (promotion_rank) {
                add_promotion_moves(from, forward, false, moves);
            } else {
                Move move;
                move.from = from;
                move.to = forward;
                move.flags = MoveFlag::Quiet;
                moves.push_back(move);

                int double_forward = from + (us == Color::White ? 16 : -16);
                bool double_rank = (us == Color::White) ? (rank == 1) : (rank == 6);
                if (double_rank && board.piece_type_at(double_forward) == PieceType::None) {
                    Move dbl = move;
                    dbl.to = double_forward;
                    dbl.flags = MoveFlag::DoublePush;
                    moves.push_back(dbl);
                }
            }
        }

        Bitboard attacks = pawn_attacks(us, from) & enemy;
        while (attacks) {
            int to = pop_lsb(attacks);
            bool promotion_rank = (us == Color::White) ? (rank == 6) : (rank == 1);
            if (promotion_rank) {
                add_promotion_moves(from, to, true, moves);
            } else {
                Move move;
                move.from = from;
                move.to = to;
                move.flags = MoveFlag::Capture;
                moves.push_back(move);
            }
        }

        int ep_square = board.en_passant_square();
        if (ep_square != -1 && (pawn_attacks(us, from) & square_bb(static_cast<Square>(ep_square)))) {
            Move move;
            move.from = from;
            move.to = ep_square;
            move.flags = MoveFlag::Capture | MoveFlag::EnPassant;
            moves.push_back(move);
        }
    }

    // Knight moves
    Bitboard knights = board.pieces(us, PieceType::Knight);
    while (knights) {
        int from = pop_lsb(knights);
        Bitboard targets = knight_attacks(from) & ~friendly;
        while (targets) {
            int to = pop_lsb(targets);
            Move move;
            move.from = from;
            move.to = to;
            move.flags = (enemy & square_bb(static_cast<Square>(to))) ? MoveFlag::Capture : MoveFlag::Quiet;
            moves.push_back(move);
        }
    }

    // Bishop moves
    Bitboard bishops = board.pieces(us, PieceType::Bishop);
    while (bishops) {
        int from = pop_lsb(bishops);
        Bitboard targets = bishop_attacks(from, occupied) & ~friendly;
        while (targets) {
            int to = pop_lsb(targets);
            Move move;
            move.from = from;
            move.to = to;
            move.flags = (enemy & square_bb(static_cast<Square>(to))) ? MoveFlag::Capture : MoveFlag::Quiet;
            moves.push_back(move);
        }
    }

    // Rook moves
    Bitboard rooks = board.pieces(us, PieceType::Rook);
    while (rooks) {
        int from = pop_lsb(rooks);
        Bitboard targets = rook_attacks(from, occupied) & ~friendly;
        while (targets) {
            int to = pop_lsb(targets);
            Move move;
            move.from = from;
            move.to = to;
            move.flags = (enemy & square_bb(static_cast<Square>(to))) ? MoveFlag::Capture : MoveFlag::Quiet;
            moves.push_back(move);
        }
    }

    // Queen moves
    Bitboard queens = board.pieces(us, PieceType::Queen);
    while (queens) {
        int from = pop_lsb(queens);
        Bitboard targets = queen_attacks(from, occupied) & ~friendly;
        while (targets) {
            int to = pop_lsb(targets);
            Move move;
            move.from = from;
            move.to = to;
            move.flags = (enemy & square_bb(static_cast<Square>(to))) ? MoveFlag::Capture : MoveFlag::Quiet;
            moves.push_back(move);
        }
    }

    // King moves
    Bitboard kings = board.pieces(us, PieceType::King);
    if (kings) {
        int from = pop_lsb(kings);
        Bitboard targets = king_attacks(from) & ~friendly;
        while (targets) {
            int to = pop_lsb(targets);
            Move move;
            move.from = from;
            move.to = to;
            move.flags = (enemy & square_bb(static_cast<Square>(to))) ? MoveFlag::Capture : MoveFlag::Quiet;
            moves.push_back(move);
        }

        // Castling
        if (!board.in_check(us)) {
            std::uint8_t rights = board.castling_rights();
            if (us == Color::White) {
                if ((rights & kWhiteKingCastle) &&
                    board.piece_type_at(static_cast<int>(Square::F1)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::G1)) == PieceType::None &&
                    !board.is_square_attacked(Square::F1, them) &&
                    !board.is_square_attacked(Square::G1, them)) {
                    Move move;
                    move.from = from;
                    move.to = static_cast<int>(Square::G1);
                    move.flags = MoveFlag::KingCastle;
                    moves.push_back(move);
                }
                if ((rights & kWhiteQueenCastle) &&
                    board.piece_type_at(static_cast<int>(Square::D1)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::C1)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::B1)) == PieceType::None &&
                    !board.is_square_attacked(Square::D1, them) &&
                    !board.is_square_attacked(Square::C1, them)) {
                    Move move;
                    move.from = from;
                    move.to = static_cast<int>(Square::C1);
                    move.flags = MoveFlag::QueenCastle;
                    moves.push_back(move);
                }
            } else {
                if ((rights & kBlackKingCastle) &&
                    board.piece_type_at(static_cast<int>(Square::F8)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::G8)) == PieceType::None &&
                    !board.is_square_attacked(Square::F8, them) &&
                    !board.is_square_attacked(Square::G8, them)) {
                    Move move;
                    move.from = from;
                    move.to = static_cast<int>(Square::G8);
                    move.flags = MoveFlag::KingCastle;
                    moves.push_back(move);
                }
                if ((rights & kBlackQueenCastle) &&
                    board.piece_type_at(static_cast<int>(Square::D8)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::C8)) == PieceType::None &&
                    board.piece_type_at(static_cast<int>(Square::B8)) == PieceType::None &&
                    !board.is_square_attacked(Square::D8, them) &&
                    !board.is_square_attacked(Square::C8, them)) {
                    Move move;
                    move.from = from;
                    move.to = static_cast<int>(Square::C8);
                    move.flags = MoveFlag::QueenCastle;
                    moves.push_back(move);
                }
            }
        }
    }
}

void MoveGenerator::generate_legal_moves(Board& board, std::vector<Move>& moves) {
    std::vector<Move> pseudo;
    pseudo.reserve(64);
    generate_pseudo_legal_moves(board, pseudo);

    moves.clear();
    moves.reserve(pseudo.size());

    for (const Move& move : pseudo) {
        Board::State state;
        board.make_move(move, state);
        if (!board.in_check(opposite_color(board.side_to_move()))) {
            moves.push_back(move);
        }
        board.undo_move(move, state);
    }
}

std::vector<Move> MoveGenerator::generate_legal_moves(Board& board) {
    std::vector<Move> moves;
    generate_legal_moves(board, moves);
    return moves;
}

void MoveGenerator::add_promotion_moves(int from, int to, bool is_capture, std::vector<Move>& moves) {
    static constexpr PieceType kPromotions[] = {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight};
    for (PieceType promotion : kPromotions) {
        Move move;
        move.from = from;
        move.to = to;
        move.promotion = promotion;
        move.flags = MoveFlag::Promotion;
        if (is_capture) {
            move.flags |= MoveFlag::Capture;
        }
        moves.push_back(move);
    }
}

}  // namespace chiron

