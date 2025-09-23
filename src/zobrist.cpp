#include "zobrist.h"

#include <random>

namespace chiron {

namespace {
std::uint64_t random_64(std::mt19937_64& rng) {
    return rng();
}
}  // namespace

void Zobrist::init() {
    if (initialized_) {
        return;
    }
    std::mt19937_64 rng(0x434849524f4eULL);  // "CHIRON" in ASCII hex for determinism.

    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            for (int sq = 0; sq < kBoardSize; ++sq) {
                pieces_[color][piece][sq] = random_64(rng);
            }
        }
    }

    for (std::uint8_t rights = 0; rights < 16; ++rights) {
        castling_[rights] = random_64(rng);
    }

    for (int file = 0; file < 8; ++file) {
        en_passant_[file] = random_64(rng);
    }

    side_ = random_64(rng);
    initialized_ = true;
}

std::uint64_t Zobrist::piece_key(Color c, PieceType pt, int square) {
    init();
    if (pt == PieceType::None || square < 0 || square >= kBoardSize) {
        return 0ULL;
    }
    return pieces_[static_cast<int>(c)][static_cast<int>(pt)][square];
}

std::uint64_t Zobrist::castling_key(std::uint8_t rights) {
    init();
    return castling_[rights & 0x0F];
}

std::uint64_t Zobrist::en_passant_key(int file) {
    init();
    if (file < 0 || file >= 8) {
        return 0ULL;
    }
    return en_passant_[file];
}

std::uint64_t Zobrist::side_key() {
    init();
    return side_;
}

}  // namespace chiron

