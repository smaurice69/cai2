#include "nnue/evaluator.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>

#include "bitboard.h"

namespace chiron::nnue {

Evaluator::Evaluator() = default;

void Evaluator::set_network_path(std::string path) {

    std::scoped_lock guard(load_mutex_);

    network_path_ = std::move(path);
    network_loaded_.store(false, std::memory_order_relaxed);
}

void Evaluator::ensure_network_loaded() const {
    if (network_loaded_.load(std::memory_order_acquire)) {
        return;
    }

    std::scoped_lock guard(load_mutex_);
    if (network_loaded_.load(std::memory_order_acquire)) {

        return;
    }
    try {
        if (!network_path_.empty()) {
            network_.load_from_file(network_path_);
        } else {
            network_.load_default();
        }
    } catch (const std::exception& ex) {
        std::cerr << "info string NNUE fallback: " << ex.what() << std::endl;
        network_.load_default();
    }
    network_loaded_.store(true, std::memory_order_release);
}

void Evaluator::apply_feature(Accumulator& accum, Color color, PieceType piece, int square, int sign) const {
    if (piece == PieceType::None || square < 0 || square >= kBoardSize) {
        return;
    }
    const Network& net = network();
    std::size_t hidden = net.hidden_size();
    if (accum.white.size() != hidden || accum.black.size() != hidden) {
        accum.reset(hidden);
    }
    std::size_t feature = feature_index(color, piece, square);
    for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
        int32_t weight = net.input_weight(feature, neuron);
        if (color == Color::White) {
            accum.white[neuron] += sign * weight;
        } else {
            accum.black[neuron] += sign * weight;
        }
    }
}

void Evaluator::build_accumulator(const Board& board, Accumulator& accum) const {
    ensure_network_loaded();
    const Network& net = network();
    accum.reset(net.hidden_size());
    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
            while (bb) {
                int square = pop_lsb(bb);
                apply_feature(accum, static_cast<Color>(color), static_cast<PieceType>(piece), square, +1);
            }
        }
    }
}

void Evaluator::update_accumulator(const Board& board, const Move& move, const Accumulator& base,
                                   Accumulator& dest) const {
    ensure_network_loaded();
    dest = base;

    Color us = board.side_to_move();
    PieceType moving_piece = board.piece_type_at(move.from);
    if (moving_piece == PieceType::None) {
        return;
    }

    apply_feature(dest, us, moving_piece, move.from, -1);

    PieceType placed_piece = moving_piece;
    if (move.is_promotion()) {
        placed_piece = move.promotion;
    }
    apply_feature(dest, us, placed_piece, move.to, +1);

    if (move.is_capture()) {
        Color them = opposite_color(us);
        int capture_square = move.to;
        PieceType captured_piece = move.is_en_passant() ? PieceType::Pawn : board.piece_type_at(move.to);
        if (move.is_en_passant()) {
            capture_square += (us == Color::White ? -8 : 8);
        }
        apply_feature(dest, them, captured_piece, capture_square, -1);
    }

    if (move.is_castle()) {
        int rook_from = 0;
        int rook_to = 0;
        if (move.flags & MoveFlag::KingCastle) {
            rook_from = (us == Color::White) ? static_cast<int>(Square::H1) : static_cast<int>(Square::H8);
            rook_to = (us == Color::White) ? static_cast<int>(Square::F1) : static_cast<int>(Square::F8);
        } else {
            rook_from = (us == Color::White) ? static_cast<int>(Square::A1) : static_cast<int>(Square::A8);
            rook_to = (us == Color::White) ? static_cast<int>(Square::D1) : static_cast<int>(Square::D8);
        }
        apply_feature(dest, us, PieceType::Rook, rook_from, -1);
        apply_feature(dest, us, PieceType::Rook, rook_to, +1);
    }
}

int Evaluator::evaluate(const Board& board, const Accumulator& accum) const {
    ensure_network_loaded();
    const Network& net = network();
    std::size_t hidden = net.hidden_size();
    double raw = static_cast<double>(net.bias());
    for (std::size_t neuron = 0; neuron < hidden; ++neuron) {
        int32_t pre = 0;
        if (neuron < accum.white.size()) {
            pre += accum.white[neuron];
        }
        if (neuron < accum.black.size()) {
            pre -= accum.black[neuron];
        }
        pre += net.hidden_bias(neuron);
        double normalized = static_cast<double>(pre) / kActivationScale;
        double activation = std::tanh(normalized) * kActivationScale;
        raw += activation * static_cast<double>(net.output_weight(neuron));
    }
    double scaled = raw * static_cast<double>(net.scale());
    int score = static_cast<int>(std::llround(scaled));
    score = std::clamp(score, -kMaxEvaluationMagnitude, kMaxEvaluationMagnitude);
    return board.side_to_move() == Color::White ? score : -score;
}

const Network& Evaluator::network() const {
    ensure_network_loaded();
    return network_;
}

}  // namespace chiron::nnue

