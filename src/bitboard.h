#pragma once

#include <bit>
#include <cstdint>
#include <ostream>

#include "types.h"

namespace chiron {

using Bitboard = std::uint64_t;

constexpr inline Bitboard kOne = 1ULL;
constexpr inline Bitboard kEmpty = 0ULL;

/**
 * @brief Returns a bitboard with the bit at @p sq set.
 */
constexpr inline Bitboard square_bb(Square sq) {
    return kOne << static_cast<int>(sq);
}

/**
 * @brief Counts the number of set bits in the bitboard.
 */
constexpr inline int popcount(Bitboard b) {
    return std::popcount(b);
}

/**
 * @brief Tests whether the bitboard has a bit set at @p sq.
 */
constexpr inline bool contains(Bitboard b, Square sq) {
    return (b & square_bb(sq)) != 0ULL;
}

/**
 * @brief Removes and returns the index of the least significant bit set.
 *
 * This helper is used to iterate through bitboards efficiently.
 */
inline int pop_lsb(Bitboard& b) {
    int idx = std::countr_zero(b);
    b &= b - 1;
    return idx;
}

/**
 * @brief Converts a square to its file (0 = file 'a').
 */
constexpr inline int file_of(Square sq) {
    return static_cast<int>(sq) & 7;
}

/**
 * @brief Converts a square to its rank (0 = rank '1').
 */
constexpr inline int rank_of(Square sq) {
    return static_cast<int>(sq) >> 3;
}

/**
 * @brief Returns a human-readable algebraic coordinate string for the square.
 */
inline std::string square_to_string(Square sq) {
    if (sq == Square::None) {
        return "-";
    }
    char file = static_cast<char>('a' + file_of(sq));
    char rank = static_cast<char>('1' + rank_of(sq));
    return std::string{file, rank};
}

/**
 * @brief Shifts a bitboard north (toward rank 8).
 */
constexpr inline Bitboard north(Bitboard b) { return b << 8; }

/**
 * @brief Shifts a bitboard south (toward rank 1).
 */
constexpr inline Bitboard south(Bitboard b) { return b >> 8; }

/**
 * @brief Shifts a bitboard east (toward file 'h').
 */
constexpr inline Bitboard east(Bitboard b) { return (b & 0x7f7f7f7f7f7f7f7fULL) << 1; }

/**
 * @brief Shifts a bitboard west (toward file 'a').
 */
constexpr inline Bitboard west(Bitboard b) { return (b & 0xfefefefefefefefeULL) >> 1; }

/**
 * @brief Shifts a bitboard north-east.
 */
constexpr inline Bitboard north_east(Bitboard b) { return (b & 0x7f7f7f7f7f7f7f7fULL) << 9; }

/**
 * @brief Shifts a bitboard north-west.
 */
constexpr inline Bitboard north_west(Bitboard b) { return (b & 0xfefefefefefefefeULL) << 7; }

/**
 * @brief Shifts a bitboard south-east.
 */
constexpr inline Bitboard south_east(Bitboard b) { return (b & 0x7f7f7f7f7f7f7f7fULL) >> 7; }

/**
 * @brief Shifts a bitboard south-west.
 */
constexpr inline Bitboard south_west(Bitboard b) { return (b & 0xfefefefefefefefeULL) >> 9; }

/**
 * @brief Convenience stream operator for printing bitboards during debugging.
 */
std::ostream& operator<<(std::ostream& os, Bitboard b);

}  // namespace chiron

