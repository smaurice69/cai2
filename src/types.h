#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace chiron {

/**
 * @brief Enumeration of chess piece colors.
 */
enum class Color : std::uint8_t {
    White = 0,
    Black = 1
};

/**
 * @brief Enumeration of chess piece types.
 */
enum class PieceType : std::uint8_t {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    None
};

/**
 * @brief Enumeration of board squares (0 = A1, 63 = H8).
 */
enum class Square : std::uint8_t {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    None = 64
};

constexpr inline int kBoardSize = 64;
constexpr inline int kNumPieceTypes = 6;
constexpr inline int kNumColors = 2;

constexpr inline Color opposite_color(Color c) {
    return c == Color::White ? Color::Black : Color::White;
}

constexpr inline std::string color_to_string(Color c) {
    return c == Color::White ? "white" : "black";
}

}  // namespace chiron

