#pragma once

#include <cstdint>
#include <string>

#include "types.h"

namespace chiron {

/**
 * @brief Bit-flags describing move characteristics.
 */
enum MoveFlag : std::uint8_t {
    Quiet = 0,
    Capture = 1 << 0,
    DoublePush = 1 << 1,
    KingCastle = 1 << 2,
    QueenCastle = 1 << 3,
    EnPassant = 1 << 4,
    Promotion = 1 << 5
};

/**
 * @brief Encodes a chess move with optional promotion information.
 */
struct Move {
    int from = 0;
    int to = 0;
    PieceType promotion = PieceType::None;
    std::uint8_t flags = MoveFlag::Quiet;

    bool is_capture() const { return flags & MoveFlag::Capture; }
    bool is_double_pawn_push() const { return flags & MoveFlag::DoublePush; }
    bool is_en_passant() const { return flags & MoveFlag::EnPassant; }
    bool is_castle() const { return flags & (MoveFlag::KingCastle | MoveFlag::QueenCastle); }
    bool is_promotion() const { return flags & MoveFlag::Promotion; }
};

inline std::string move_to_string(const Move& m) {
    std::string str;
    str.reserve(5);
    str += static_cast<char>('a' + (m.from & 7));
    str += static_cast<char>('1' + (m.from >> 3));
    str += static_cast<char>('a' + (m.to & 7));
    str += static_cast<char>('1' + (m.to >> 3));
    if (m.is_promotion()) {
        switch (m.promotion) {
            case PieceType::Knight:
                str += 'n';
                break;
            case PieceType::Bishop:
                str += 'b';
                break;
            case PieceType::Rook:
                str += 'r';
                break;
            case PieceType::Queen:
            default:
                str += 'q';
                break;
        }
    }
    return str;
}

}  // namespace chiron

