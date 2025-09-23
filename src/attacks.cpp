#include "attacks.h"

#include <array>

namespace chiron {

namespace {
std::array<std::array<Bitboard, kBoardSize>, kNumColors> pawn_attacks_{};
std::array<Bitboard, kBoardSize> knight_attacks_{};
std::array<Bitboard, kBoardSize> king_attacks_{};
bool initialized = false;

Bitboard mask_knight(int square) {
    Bitboard attacks = kEmpty;
    int rank = square / 8;
    int file = square % 8;
    auto add = [&](int r, int f) {
        if (r >= 0 && r < 8 && f >= 0 && f < 8) {
            attacks |= square_bb(static_cast<Square>(r * 8 + f));
        }
    };
    add(rank + 2, file + 1);
    add(rank + 2, file - 1);
    add(rank - 2, file + 1);
    add(rank - 2, file - 1);
    add(rank + 1, file + 2);
    add(rank + 1, file - 2);
    add(rank - 1, file + 2);
    add(rank - 1, file - 2);
    return attacks;
}

Bitboard mask_king(int square) {
    Bitboard b = square_bb(static_cast<Square>(square));
    Bitboard attacks = kEmpty;
    attacks |= north(b);
    attacks |= south(b);
    attacks |= east(b);
    attacks |= west(b);
    attacks |= north_east(b);
    attacks |= north_west(b);
    attacks |= south_east(b);
    attacks |= south_west(b);
    return attacks;
}

Bitboard mask_pawn(Color color, int square) {
    Bitboard b = square_bb(static_cast<Square>(square));
    if (color == Color::White) {
        return north_east(b) | north_west(b);
    }
    return south_east(b) | south_west(b);
}

}  // namespace

void init_attack_tables() {
    if (initialized) {
        return;
    }

    for (int sq = 0; sq < kBoardSize; ++sq) {
        knight_attacks_[sq] = mask_knight(sq);
        king_attacks_[sq] = mask_king(sq);
        pawn_attacks_[static_cast<int>(Color::White)][sq] = mask_pawn(Color::White, sq);
        pawn_attacks_[static_cast<int>(Color::Black)][sq] = mask_pawn(Color::Black, sq);
    }
    initialized = true;
}

Bitboard pawn_attacks(Color color, int square) {
    init_attack_tables();
    return pawn_attacks_[static_cast<int>(color)][square];
}

Bitboard knight_attacks(int square) {
    init_attack_tables();
    return knight_attacks_[square];
}

Bitboard king_attacks(int square) {
    init_attack_tables();
    return king_attacks_[square];
}

Bitboard bishop_attacks(int square, Bitboard blockers) {
    Bitboard attacks = kEmpty;
    int rank = square / 8;
    int file = square % 8;

    for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; ++r, ++f) {
        int sq = r * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; ++r, --f) {
        int sq = r * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; --r, ++f) {
        int sq = r * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; --r, --f) {
        int sq = r * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    return attacks;
}

Bitboard rook_attacks(int square, Bitboard blockers) {
    Bitboard attacks = kEmpty;
    int rank = square / 8;
    int file = square % 8;

    for (int r = rank + 1; r <= 7; ++r) {
        int sq = r * 8 + file;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int r = rank - 1; r >= 0; --r) {
        int sq = r * 8 + file;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int f = file + 1; f <= 7; ++f) {
        int sq = rank * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    for (int f = file - 1; f >= 0; --f) {
        int sq = rank * 8 + f;
        attacks |= square_bb(static_cast<Square>(sq));
        if (blockers & square_bb(static_cast<Square>(sq))) {
            break;
        }
    }
    return attacks;
}

}  // namespace chiron

