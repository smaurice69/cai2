#include "bitboard.h"

#include <bitset>

namespace chiron {

std::ostream& operator<<(std::ostream& os, Bitboard b) {
    for (int rank = 7; rank >= 0; --rank) {
        os << rank + 1 << " ";
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            os << ((b & (kOne << sq)) ? "1 " : ". ");
        }
        os << '\n';
    }
    os << "  a b c d e f g h\n";
    return os;
}

}  // namespace chiron

