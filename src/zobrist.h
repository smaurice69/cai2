#pragma once

#include <array>
#include <cstdint>

#include "types.h"

namespace chiron {

/**
 * @brief Zobrist hashing helper managing random bitstrings for board states.
 */
class Zobrist {
   public:
    static void init();

    static std::uint64_t piece_key(Color c, PieceType pt, int square);
    static std::uint64_t castling_key(std::uint8_t rights);
    static std::uint64_t en_passant_key(int file);
    static std::uint64_t side_key();

   private:
    static inline bool initialized_ = false;
    static inline std::uint64_t pieces_[kNumColors][kNumPieceTypes][kBoardSize]{};
    static inline std::uint64_t castling_[16]{};
    static inline std::uint64_t en_passant_[8]{};
    static inline std::uint64_t side_ = 0ULL;
};

}  // namespace chiron

