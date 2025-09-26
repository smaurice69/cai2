#include "training/trainer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "bitboard.h"
#include "nnue/evaluator.h"

namespace chiron {

namespace {

constexpr int kWeightLimit = 40000;

int clamp_weight(int value) {
    return std::clamp(value, -kWeightLimit, kWeightLimit);
}

int evaluate_with_network(const Board& board, const nnue::Network& network) {
    std::size_t hidden = network.hidden_size();
    std::vector<int32_t> white(hidden, 0);
    std::vector<int32_t> black(hidden, 0);
    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
            while (bb) {
                int sq = pop_lsb(bb);
                std::size_t feature = nnue::feature_index(static_cast<Color>(color), static_cast<PieceType>(piece), sq);
                for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
                    int32_t weight = network.input_weight(feature, neuron);
                    if (color == static_cast<int>(Color::White)) {
                        white[neuron] += weight;
                    } else {
                        black[neuron] += weight;
                    }
                }
            }
        }
    }

    double raw = static_cast<double>(network.bias());
    for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
        int32_t pre = white[neuron] - black[neuron] + network.hidden_bias(neuron);
        double normalized = static_cast<double>(pre) / nnue::kActivationScale;
        double activation = std::tanh(normalized) * nnue::kActivationScale;
        raw += activation * static_cast<double>(network.output_weight(neuron));
    }
    double scaled = raw * static_cast<double>(network.scale());
    int eval = static_cast<int>(std::llround(scaled));
    eval = std::clamp(eval, -nnue::kMaxEvaluationMagnitude, nnue::kMaxEvaluationMagnitude);
    return board.side_to_move() == Color::White ? eval : -eval;
}

}  // namespace

ParameterSet::ParameterSet(std::size_t hidden_size) { network_.load_default(hidden_size); }

void ParameterSet::reset(std::size_t hidden_size) { network_.load_default(hidden_size); }

void ParameterSet::load(const std::string& path) { network_.load_from_file(path); }

void ParameterSet::save(const std::string& path) const {
    namespace fs = std::filesystem;
    fs::path target(path);
    fs::path temp = target;
    temp += ".tmp";

    network_.save_to_file(temp.string());

    std::error_code ec;
    fs::rename(temp, target, ec);
    if (ec) {
        fs::remove(target, ec);
        fs::rename(temp, target, ec);
        if (ec) {
            fs::remove(temp);
            throw std::runtime_error("Failed to replace NNUE network file: " + ec.message());
        }
    }
}

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

    nnue::Network& net = parameters.network();
    std::size_t hidden = net.hidden_size();
    std::vector<std::size_t> white_features;
    std::vector<std::size_t> black_features;
    white_features.reserve(32);
    black_features.reserve(32);
    std::vector<int32_t> white_accum(hidden);
    std::vector<int32_t> black_accum(hidden);
    std::vector<double> activations(hidden);
    std::vector<double> activation_derivatives(hidden);

    for (const TrainingExample& example : batch) {
        Board board;
        board.set_from_fen(example.fen);

        white_features.clear();
        black_features.clear();
        for (int color = 0; color < kNumColors; ++color) {
            for (int piece = 0; piece < kNumPieceTypes; ++piece) {
                Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
                while (bb) {
                    int square = pop_lsb(bb);
                    std::size_t feature = nnue::feature_index(static_cast<Color>(color), static_cast<PieceType>(piece), square);
                    if (color == static_cast<int>(Color::White)) {
                        white_features.push_back(feature);
                    } else {
                        black_features.push_back(feature);
                    }
                }
            }
        }

        std::fill(white_accum.begin(), white_accum.end(), 0);
        std::fill(black_accum.begin(), black_accum.end(), 0);
        for (std::size_t feature : white_features) {
            for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
                white_accum[neuron] += net.input_weight(feature, neuron);
            }
        }
        for (std::size_t feature : black_features) {
            for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
                black_accum[neuron] += net.input_weight(feature, neuron);
            }
        }

        double raw = static_cast<double>(net.bias());
        for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
            int32_t pre = white_accum[neuron] - black_accum[neuron] + net.hidden_bias(neuron);
            double normalized = static_cast<double>(pre) / nnue::kActivationScale;
            double tanh_val = std::tanh(normalized);
            activations[neuron] = tanh_val * nnue::kActivationScale;
            activation_derivatives[neuron] = 1.0 - tanh_val * tanh_val;
            raw += activations[neuron] * static_cast<double>(net.output_weight(neuron));
        }

        double orientation = (board.side_to_move() == Color::White) ? 1.0 : -1.0;
        double predicted_cp = orientation * raw * static_cast<double>(net.scale());
        double error = static_cast<double>(example.target_cp) - predicted_cp;
        double lr_error = config_.learning_rate * error * orientation * static_cast<double>(net.scale());

        double bias_current = static_cast<double>(net.bias());
        double bias_next = bias_current + lr_error;
        if (config_.regularisation > 0.0) {
            bias_next -= config_.regularisation * bias_current;
        }
        net.set_bias(clamp_weight(static_cast<int>(std::llround(bias_next))));

        for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
            double output_current = static_cast<double>(net.output_weight(neuron));
            double output_next = output_current + lr_error * activations[neuron];
            if (config_.regularisation > 0.0) {
                output_next -= config_.regularisation * output_current;
            }
            net.set_output_weight(neuron, static_cast<float>(output_next));

            double grad_pre = lr_error * output_current * activation_derivatives[neuron];
            int32_t hidden_bias_current = net.hidden_bias(neuron);
            double hidden_next = static_cast<double>(hidden_bias_current) + grad_pre;
            if (config_.regularisation > 0.0) {
                hidden_next -= config_.regularisation * static_cast<double>(hidden_bias_current);
            }
            net.set_hidden_bias(neuron, clamp_weight(static_cast<int>(std::llround(hidden_next))));

            if (std::abs(grad_pre) < 1e-12) {
                continue;
            }

            for (std::size_t feature : white_features) {
                int32_t current = net.input_weight(feature, neuron);
                double next = static_cast<double>(current) + grad_pre;
                if (config_.regularisation > 0.0) {
                    next -= config_.regularisation * static_cast<double>(current);
                }
                net.set_input_weight(feature, neuron,
                                     clamp_weight(static_cast<int>(std::llround(next))));
            }
            for (std::size_t feature : black_features) {
                int32_t current = net.input_weight(feature, neuron);
                double next = static_cast<double>(current) - grad_pre;
                if (config_.regularisation > 0.0) {
                    next -= config_.regularisation * static_cast<double>(current);
                }
                net.set_input_weight(feature, neuron,
                                     clamp_weight(static_cast<int>(std::llround(next))));
            }
        }
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

