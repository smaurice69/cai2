#include "network.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace chiron::nnue {

namespace {

constexpr char kMagic[4] = {'N', 'N', 'U', 'E'};
constexpr std::uint32_t kVersionV1 = 1U;
constexpr std::uint32_t kVersionV2 = 2U;

constexpr int kDefaultPieceValues[static_cast<int>(PieceType::King) + 1] = {
    100, 320, 330, 500, 900, 20000};

std::size_t weight_offset(std::size_t feature, std::size_t neuron) {
    return neuron * kFeatureCount + feature;
}

}  // namespace

std::size_t feature_index(Color color, PieceType piece, int square) {
    if (piece == PieceType::None) {
        throw std::invalid_argument("feature_index called with PieceType::None");
    }
    if (square < 0 || square >= kBoardSize) {
        throw std::out_of_range("Square index out of range for feature_index");
    }
    std::size_t color_offset = static_cast<std::size_t>(color) * kNumPieceTypes * kBoardSize;
    std::size_t piece_offset = static_cast<std::size_t>(piece) * kBoardSize;
    return color_offset + piece_offset + static_cast<std::size_t>(square);
}

Network::Network() = default;

void Network::ensure_storage(std::size_t hidden_size) {
    hidden_size_ = std::max<std::size_t>(1, hidden_size);
    input_weights_.assign(hidden_size_ * kFeatureCount, 0);
    hidden_biases_.assign(hidden_size_, 0);
    output_weights_.assign(hidden_size_, 0.0F);
}

void Network::set_hidden_size(std::size_t hidden_size) {
    ensure_storage(hidden_size);
    loaded_ = true;
}

void Network::load_from_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open NNUE network file: " + path);
    }

    char magic[4];
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Invalid NNUE network file: magic mismatch");
    }

    std::uint32_t version = 0;
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE network version");
    }

    std::uint32_t feature_count = 0;
    stream.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE feature count");
    }
    if (feature_count != kFeatureCount) {
        throw std::runtime_error("Unexpected feature count in NNUE network file");
    }

    if (version == kVersionV1) {
        std::int32_t bias = 0;
        float scale = 1.0F;
        stream.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        stream.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        if (!stream) {
            throw std::runtime_error("Failed to read NNUE network parameters");
        }

        std::vector<int16_t> buffer(feature_count);
        stream.read(reinterpret_cast<char*>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size() * sizeof(int16_t)));
        if (!stream) {
            throw std::runtime_error("Failed to read NNUE weights from file: " + path);
        }

        ensure_storage(1);
        for (std::size_t i = 0; i < buffer.size(); ++i) {
            input_weights_[i] = static_cast<int32_t>(buffer[i]);
        }
        hidden_biases_.assign(hidden_size_, 0);
        output_weights_.assign(hidden_size_, 1.0F);
        bias_ = bias;
        scale_ = scale;
        loaded_ = true;
        return;
    }

    if (version != kVersionV2) {
        throw std::runtime_error("Unsupported NNUE network version: " + std::to_string(version));
    }

    std::uint32_t hidden_size = 0;
    stream.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE hidden size");
    }

    std::int32_t bias = 0;
    float scale = 1.0F;
    stream.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    stream.read(reinterpret_cast<char*>(&scale), sizeof(scale));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE network parameters");
    }

    ensure_storage(hidden_size);

    std::vector<int16_t> bias_buffer(hidden_size_);
    stream.read(reinterpret_cast<char*>(bias_buffer.data()),
                static_cast<std::streamsize>(bias_buffer.size() * sizeof(int16_t)));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE hidden biases");
    }

    std::vector<float> output_buffer(hidden_size_);
    stream.read(reinterpret_cast<char*>(output_buffer.data()),
                static_cast<std::streamsize>(output_buffer.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE output weights");
    }

    std::vector<int16_t> weights_buffer(hidden_size_ * kFeatureCount);
    stream.read(reinterpret_cast<char*>(weights_buffer.data()),
                static_cast<std::streamsize>(weights_buffer.size() * sizeof(int16_t)));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE weights from file: " + path);
    }

    for (std::size_t i = 0; i < hidden_size_; ++i) {
        hidden_biases_[i] = static_cast<int32_t>(bias_buffer[i]);
        output_weights_[i] = output_buffer[i];
    }
    for (std::size_t i = 0; i < weights_buffer.size(); ++i) {
        input_weights_[i] = static_cast<int32_t>(weights_buffer[i]);
    }

    bias_ = bias;
    scale_ = scale;
    loaded_ = true;
}

void Network::load_default(std::size_t hidden_size) {
    ensure_storage(hidden_size);
    std::fill(hidden_biases_.begin(), hidden_biases_.end(), 0);
    float output = hidden_size_ > 0 ? 1.0F / static_cast<float>(hidden_size_) : 1.0F;
    std::fill(output_weights_.begin(), output_weights_.end(), output);

    for (std::size_t neuron = 0; neuron < hidden_size_; ++neuron) {
        for (int color = 0; color < kNumColors; ++color) {
            for (int piece = 0; piece < kNumPieceTypes; ++piece) {
                PieceType type = static_cast<PieceType>(piece);
                int value = kDefaultPieceValues[piece];
                for (int square = 0; square < kBoardSize; ++square) {
                    std::size_t feature = feature_index(static_cast<Color>(color), type, square);
                    input_weights_[weight_offset(feature, neuron)] = value;
                }
            }
        }
    }

    bias_ = 0;
    scale_ = 1.0F;
    loaded_ = true;
}

void Network::save_to_file(const std::string& path) const {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Failed to open NNUE network for writing: " + path);
    }

    stream.write(kMagic, sizeof(kMagic));
    std::uint32_t version = kVersionV2;
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    std::uint32_t feature_count = static_cast<std::uint32_t>(kFeatureCount);
    stream.write(reinterpret_cast<const char*>(&feature_count), sizeof(feature_count));
    std::uint32_t hidden = static_cast<std::uint32_t>(hidden_size_);
    stream.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
    stream.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
    stream.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));

    std::vector<int16_t> bias_buffer(hidden_size_);
    for (std::size_t i = 0; i < hidden_size_; ++i) {
        int32_t value = hidden_biases_[i];
        value = std::clamp(value, static_cast<int32_t>(-32768), static_cast<int32_t>(32767));
        bias_buffer[i] = static_cast<int16_t>(value);
    }
    stream.write(reinterpret_cast<const char*>(bias_buffer.data()),
                 static_cast<std::streamsize>(bias_buffer.size() * sizeof(int16_t)));

    stream.write(reinterpret_cast<const char*>(output_weights_.data()),
                 static_cast<std::streamsize>(output_weights_.size() * sizeof(float)));

    std::vector<int16_t> weights_buffer(hidden_size_ * kFeatureCount);
    for (std::size_t i = 0; i < weights_buffer.size(); ++i) {
        int32_t value = input_weights_[i];
        value = std::clamp(value, static_cast<int32_t>(-32768), static_cast<int32_t>(32767));
        weights_buffer[i] = static_cast<int16_t>(value);
    }
    stream.write(reinterpret_cast<const char*>(weights_buffer.data()),
                 static_cast<std::streamsize>(weights_buffer.size() * sizeof(int16_t)));

    if (!stream) {
        throw std::runtime_error("Failed to write NNUE network file: " + path);
    }
}

int32_t Network::input_weight(Color color, PieceType piece, int square, std::size_t neuron) const {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize || neuron >= hidden_size_) {
        return 0;
    }
    std::size_t feature = feature_index(color, piece, square);
    return input_weights_[weight_offset(feature, neuron)];
}

int32_t Network::input_weight(std::size_t feature, std::size_t neuron) const {
    if (neuron >= hidden_size_ || feature >= kFeatureCount) {
        return 0;
    }
    return input_weights_[weight_offset(feature, neuron)];
}

void Network::set_input_weight(Color color, PieceType piece, int square, int32_t value, std::size_t neuron) {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize || neuron >= hidden_size_) {
        return;
    }
    std::size_t feature = feature_index(color, piece, square);
    input_weights_[weight_offset(feature, neuron)] = value;
    loaded_ = true;
}

void Network::set_input_weight(std::size_t feature, std::size_t neuron, int32_t value) {
    if (feature >= kFeatureCount || neuron >= hidden_size_) {
        return;
    }
    input_weights_[weight_offset(feature, neuron)] = value;
    loaded_ = true;
}

void Network::add_input_weight(Color color, PieceType piece, int square, int32_t delta, std::size_t neuron) {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize || neuron >= hidden_size_) {
        return;
    }
    std::size_t feature = feature_index(color, piece, square);
    input_weights_[weight_offset(feature, neuron)] += delta;
    loaded_ = true;
}

void Network::add_input_weight(std::size_t feature, std::size_t neuron, int32_t delta) {
    if (feature >= kFeatureCount || neuron >= hidden_size_) {
        return;
    }
    input_weights_[weight_offset(feature, neuron)] += delta;
    loaded_ = true;
}

int32_t Network::hidden_bias(std::size_t neuron) const {
    if (neuron >= hidden_size_) {
        return 0;
    }
    return hidden_biases_[neuron];
}

void Network::set_hidden_bias(std::size_t neuron, int32_t value) {
    if (neuron >= hidden_size_) {
        return;
    }
    hidden_biases_[neuron] = value;
    loaded_ = true;
}

float Network::output_weight(std::size_t neuron) const {
    if (neuron >= hidden_size_) {
        return 0.0F;
    }
    return output_weights_[neuron];
}

void Network::set_output_weight(std::size_t neuron, float value) {
    if (neuron >= hidden_size_) {
        return;
    }
    output_weights_[neuron] = value;
    loaded_ = true;
}

void Network::set_bias(int32_t bias) {
    bias_ = bias;
    loaded_ = true;
}

void Network::set_scale(float scale) {
    scale_ = scale;
    loaded_ = true;
}

}  // namespace chiron::nnue

