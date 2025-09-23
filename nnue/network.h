#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "types.h"

namespace chiron::nnue {

constexpr std::size_t kFeatureCount = static_cast<std::size_t>(kNumColors) * static_cast<std::size_t>(kNumPieceTypes) *
                                       static_cast<std::size_t>(kBoardSize);

/**
 * @brief Returns the index into the flattened feature array for a piece on a square.
 */
std::size_t feature_index(Color color, PieceType piece, int square);

/**
 * @brief Represents a compact NNUE-style network with a single accumulator layer.
 *
 * The network stores weights for each (color, piece type, square) feature and a bias/scale
 * used to convert accumulated sums into centipawn evaluations.
 */
class Network {
   public:
    Network();

    void load_from_file(const std::string& path);
    void load_default();

    [[nodiscard]] bool is_loaded() const { return loaded_; }

    [[nodiscard]] int32_t weight(Color color, PieceType piece, int square) const;
    [[nodiscard]] int32_t bias() const { return bias_; }
    [[nodiscard]] float scale() const { return scale_; }

   private:
    bool loaded_ = false;
    std::array<int32_t, kFeatureCount> weights_{};
    int32_t bias_ = 0;
    float scale_ = 1.0f;
};

}  // namespace chiron::nnue

