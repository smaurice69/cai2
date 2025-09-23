#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bitboard.h"
#include "move.h"
#include "types.h"

namespace chiron {

constexpr inline std::uint8_t kWhiteKingCastle = 1 << 0;
constexpr inline std::uint8_t kWhiteQueenCastle = 1 << 1;
constexpr inline std::uint8_t kBlackKingCastle = 1 << 2;
constexpr inline std::uint8_t kBlackQueenCastle = 1 << 3;

/**
 * @brief Represents the full state of a chess board including meta information.
 */
class Board {
   public:
    struct State {
        std::uint8_t castling_rights = 0;
        int en_passant_square = -1;
        int halfmove_clock = 0;
        std::uint64_t zobrist_key = 0ULL;
        PieceType captured_piece = PieceType::None;
        int fullmove_number = 1;
    };

    Board();

    void set_start_position();
    void set_from_fen(const std::string& fen);

    [[nodiscard]] Bitboard pieces(Color color, PieceType type) const {
        return pieces_[static_cast<int>(color)][static_cast<int>(type)];
    }

    [[nodiscard]] Bitboard occupancy(Color color) const {
        return occupancies_[static_cast<int>(color)];
    }

    [[nodiscard]] Bitboard occupancy_all() const { return occupancy_all_; }

    [[nodiscard]] Color side_to_move() const { return side_to_move_; }
    [[nodiscard]] std::uint8_t castling_rights() const { return castling_rights_; }
    [[nodiscard]] int en_passant_square() const { return en_passant_square_; }
    [[nodiscard]] int halfmove_clock() const { return halfmove_clock_; }
    [[nodiscard]] int fullmove_number() const { return fullmove_number_; }
    [[nodiscard]] std::uint64_t zobrist_key() const { return zobrist_key_; }

    [[nodiscard]] PieceType piece_type_at(int square) const;
    [[nodiscard]] std::optional<Color> color_at(int square) const;

    bool is_square_attacked(Square sq, Color by) const;
    bool in_check(Color color) const;

    void make_move(const Move& move, State& out_state);
    void undo_move(const Move& move, const State& state);

    std::string fen() const;

   private:
    void clear();
    void place_piece(Color color, PieceType type, int square);
    void remove_piece(Color color, PieceType type, int square);

    PieceType piece_from_char(char c) const;

    std::array<std::array<Bitboard, kNumPieceTypes>, kNumColors> pieces_{};
    std::array<Bitboard, kNumColors> occupancies_{};
    Bitboard occupancy_all_ = 0ULL;
    std::array<std::uint8_t, kBoardSize> mailbox_{};  // Encoded pieces per square.

    Color side_to_move_ = Color::White;
    std::uint8_t castling_rights_ = 0;
    int en_passant_square_ = -1;
    int halfmove_clock_ = 0;
    int fullmove_number_ = 1;
    std::uint64_t zobrist_key_ = 0ULL;
};

constexpr inline std::uint8_t encode_piece(Color color, PieceType type) {
    return static_cast<std::uint8_t>(static_cast<int>(type) + static_cast<int>(color) * kNumPieceTypes);
}

constexpr inline PieceType decode_piece_type(std::uint8_t code) {
    return code >= kNumPieceTypes * kNumColors ? PieceType::None
                                               : static_cast<PieceType>(code % kNumPieceTypes);
}

constexpr inline Color decode_piece_color(std::uint8_t code) {
    return static_cast<Color>(code / kNumPieceTypes);
}

constexpr inline std::uint8_t kEmptySquare = kNumPieceTypes * kNumColors;

}  // namespace chiron

