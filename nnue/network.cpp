#include "network.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace chiron::nnue {

namespace {
constexpr char kMagic[4] = {'N', 'N', 'U', 'E'};
constexpr std::uint32_t kSupportedVersion = 1U;

struct DiskHeader {
    char magic[4];
    std::uint32_t version = 0;
    std::uint32_t feature_count = 0;
    std::int32_t bias = 0;
    float scale = 1.0F;
};

constexpr int kDefaultPieceValues[static_cast<int>(PieceType::King) + 1] = {
    100, 320, 330, 500, 900, 20000};
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

void Network::load_from_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open NNUE network file: " + path);
    }

    DiskHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE network header from file: " + path);
    }
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Invalid NNUE network file: magic mismatch");
    }
    if (header.version != kSupportedVersion) {
        throw std::runtime_error("Unsupported NNUE network version: " + std::to_string(header.version));
    }
    if (header.feature_count != kFeatureCount) {
        throw std::runtime_error("Unexpected feature count in NNUE network file");
    }

    std::vector<int16_t> buffer(header.feature_count);
    stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(int16_t)));
    if (!stream) {
        throw std::runtime_error("Failed to read NNUE weights from file: " + path);
    }

    std::fill(weights_.begin(), weights_.end(), 0);
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        weights_[i] = static_cast<int32_t>(buffer[i]);
    }

    bias_ = header.bias;
    scale_ = header.scale;
    loaded_ = true;
}

void Network::load_default() {
    std::fill(weights_.begin(), weights_.end(), 0);
    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            PieceType type = static_cast<PieceType>(piece);
            int value = kDefaultPieceValues[piece];
            for (int square = 0; square < kBoardSize; ++square) {
                weights_[feature_index(static_cast<Color>(color), type, square)] = value;
            }
        }
    }
    bias_ = 0;
    scale_ = 1.0F;
    loaded_ = true;
}

void Network::save_to_file(const std::string& path) const {
    DiskHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kSupportedVersion;
    header.feature_count = static_cast<std::uint32_t>(kFeatureCount);
    header.bias = bias_;
    header.scale = scale_;

    std::vector<int16_t> buffer(kFeatureCount);
    for (std::size_t i = 0; i < kFeatureCount; ++i) {
        int32_t value = weights_[i];
        value = std::clamp(value, static_cast<int32_t>(-32768), static_cast<int32_t>(32767));
        buffer[i] = static_cast<int16_t>(value);
    }

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Failed to open NNUE network for writing: " + path);
    }
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(int16_t)));
    if (!stream) {
        throw std::runtime_error("Failed to write NNUE network file: " + path);
    }
}

int32_t Network::weight(Color color, PieceType piece, int square) const {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize) {
        return 0;
    }
    return weights_[feature_index(color, piece, square)];
}

void Network::set_weight(Color color, PieceType piece, int square, int32_t value) {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize) {
        return;
    }
    weights_[feature_index(color, piece, square)] = value;
    loaded_ = true;
}

void Network::add_weight(Color color, PieceType piece, int square, int32_t delta) {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize) {
        return;
    }
    std::size_t idx = feature_index(color, piece, square);
    weights_[idx] += delta;
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

