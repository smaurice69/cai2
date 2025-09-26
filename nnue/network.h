#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

namespace chiron::nnue {

constexpr std::size_t kFeatureCount = static_cast<std::size_t>(kNumColors) * static_cast<std::size_t>(kNumPieceTypes) *
                                       static_cast<std::size_t>(kBoardSize);
constexpr std::size_t kDefaultHiddenSize = 32;
constexpr double kActivationScale = 512.0;

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
    void load_default(std::size_t hidden_size = kDefaultHiddenSize);
    void save_to_file(const std::string& path) const;

    void set_hidden_size(std::size_t hidden_size);
    [[nodiscard]] std::size_t hidden_size() const { return hidden_size_; }
    [[nodiscard]] bool is_loaded() const { return loaded_; }

    [[nodiscard]] int32_t input_weight(Color color, PieceType piece, int square, std::size_t neuron = 0) const;
    [[nodiscard]] int32_t input_weight(std::size_t feature_index, std::size_t neuron) const;
    void set_input_weight(Color color, PieceType piece, int square, int32_t value, std::size_t neuron = 0);
    void set_input_weight(std::size_t feature_index, std::size_t neuron, int32_t value);
    void add_input_weight(Color color, PieceType piece, int square, int32_t delta, std::size_t neuron = 0);
    void add_input_weight(std::size_t feature_index, std::size_t neuron, int32_t delta);

    [[nodiscard]] int32_t hidden_bias(std::size_t neuron) const;
    void set_hidden_bias(std::size_t neuron, int32_t value);

    [[nodiscard]] float output_weight(std::size_t neuron) const;
    void set_output_weight(std::size_t neuron, float value);

    void set_bias(int32_t bias);
    void set_scale(float scale);
    [[nodiscard]] int32_t bias() const { return bias_; }
    [[nodiscard]] float scale() const { return scale_; }

    std::vector<int32_t>& input_weights_data() { return input_weights_; }
    const std::vector<int32_t>& input_weights_data() const { return input_weights_; }
    std::vector<int32_t>& hidden_biases_data() { return hidden_biases_; }
    const std::vector<int32_t>& hidden_biases_data() const { return hidden_biases_; }
    std::vector<float>& output_weights_data() { return output_weights_; }
    const std::vector<float>& output_weights_data() const { return output_weights_; }

   private:
    void ensure_storage(std::size_t hidden_size);

    bool loaded_ = false;
    std::size_t hidden_size_ = kDefaultHiddenSize;
    std::vector<int32_t> input_weights_;
    std::vector<int32_t> hidden_biases_;
    std::vector<float> output_weights_;
    int32_t bias_ = 0;
    float scale_ = 1.0f;
};

}  // namespace chiron::nnue

