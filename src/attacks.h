#pragma once

#include "bitboard.h"
#include "types.h"

namespace chiron {

void init_attack_tables();

[[nodiscard]] Bitboard pawn_attacks(Color color, int square);
[[nodiscard]] Bitboard knight_attacks(int square);
[[nodiscard]] Bitboard king_attacks(int square);
[[nodiscard]] Bitboard bishop_attacks(int square, Bitboard blockers);
[[nodiscard]] Bitboard rook_attacks(int square, Bitboard blockers);
[[nodiscard]] inline Bitboard queen_attacks(int square, Bitboard blockers) {
    return bishop_attacks(square, blockers) | rook_attacks(square, blockers);
}

}  // namespace chiron

