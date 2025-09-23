#include "training/trainer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "bitboard.h"

namespace chiron {

namespace {

constexpr int kWeightLimit = 40000;

int clamp_weight(int value) {
    return std::clamp(value, -kWeightLimit, kWeightLimit);
}

int evaluate_with_network(const Board& board, const nnue::Network& network) {
    int32_t white_sum = 0;
    int32_t black_sum = 0;
    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
            while (bb) {
                int sq = pop_lsb(bb);
                int32_t weight = network.weight(static_cast<Color>(color), static_cast<PieceType>(piece), sq);
                if (color == static_cast<int>(Color::White)) {
                    white_sum += weight;
                } else {
                    black_sum += weight;
                }
            }
        }
    }
    int32_t raw = white_sum - black_sum + network.bias();
    double scaled = static_cast<double>(raw) * static_cast<double>(network.scale());
    int eval = static_cast<int>(std::llround(scaled));
    return board.side_to_move() == Color::White ? eval : -eval;
}

}  // namespace

ParameterSet::ParameterSet() { network_.load_default(); }

void ParameterSet::load(const std::string& path) { network_.load_from_file(path); }

void ParameterSet::save(const std::string& path) const { network_.save_to_file(path); }

Trainer::Trainer() : config_({}) {}

Trainer::Trainer(Config config) : config_(config) {}

int Trainer::evaluate_example(const TrainingExample& example, const ParameterSet& parameters) const {
    Board board;
    board.set_from_fen(example.fen);
    return evaluate_with_network(board, parameters.network());
}

void Trainer::train_batch(const std::vector<TrainingExample>& batch, ParameterSet& parameters) const {
    if (batch.empty()) {
        return;
    }

    for (const TrainingExample& example : batch) {
        Board board;
        board.set_from_fen(example.fen);

        nnue::Network& net = parameters.network();
        int prediction = evaluate_with_network(board, net);
        int error = example.target_cp - prediction;
        double gradient = config_.learning_rate * static_cast<double>(error);

        for (int color = 0; color < kNumColors; ++color) {
            for (int piece = 0; piece < kNumPieceTypes; ++piece) {
                Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
                while (bb) {
                    int square = pop_lsb(bb);
                    int sign = (color == static_cast<int>(Color::White)) ? 1 : -1;
                    int32_t current = net.weight(static_cast<Color>(color), static_cast<PieceType>(piece), square);
                    double update = gradient * sign;
                    if (config_.regularisation > 0.0) {
                        update -= config_.regularisation * static_cast<double>(current);
                    }
                    int32_t next = clamp_weight(static_cast<int>(std::llround(static_cast<double>(current) + update)));
                    net.set_weight(static_cast<Color>(color), static_cast<PieceType>(piece), square, next);
                }
            }
        }

        int32_t bias = net.bias();
        double bias_update = gradient;
        if (config_.regularisation > 0.0) {
            bias_update -= config_.regularisation * static_cast<double>(bias);
        }
        net.set_bias(clamp_weight(static_cast<int>(std::llround(static_cast<double>(bias) + bias_update))));
    }
}

std::vector<TrainingExample> load_training_file(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open training data file: " + path);
    }

    std::vector<TrainingExample> data;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        auto delimiter = line.find('|');
        if (delimiter == std::string::npos) {
            continue;
        }
        TrainingExample example;
        example.fen = line.substr(0, delimiter);
        std::string score = line.substr(delimiter + 1);
        try {
            example.target_cp = std::stoi(score);
        } catch (const std::exception&) {
            continue;
        }
        data.push_back(std::move(example));
    }
    return data;
}

void save_training_file(const std::string& path, const std::vector<TrainingExample>& data) {
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Failed to open training file for writing: " + path);
    }
    for (const TrainingExample& example : data) {
        stream << example.fen << '|' << example.target_cp << '\n';
    }
}

}  // namespace chiron

