#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitboard.h"
#include "board.h"
#include "nnue/network.h"
#include "training/gpu_backend.h"

namespace chiron::gpu {

namespace {

void check_cuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error during ") + context + ": " +
                                 cudaGetErrorString(result));
    }
}

__device__ inline int clamp_weight_device(double value) {
    double rounded = nearbyint(value);
    if (rounded > static_cast<double>(kTrainerWeightLimit)) {
        rounded = static_cast<double>(kTrainerWeightLimit);
    }
    if (rounded < static_cast<double>(-kTrainerWeightLimit)) {
        rounded = static_cast<double>(-kTrainerWeightLimit);
    }
    return static_cast<int>(rounded);
}

__global__ void train_example_kernel(const int8_t* features, int target_cp, int orientation,
                                     double learning_rate, double regularisation, int hidden_size,
                                     int feature_count, int32_t* input_weights, int32_t* hidden_biases,
                                     float* output_weights, int32_t* bias, float scale) {
    extern __shared__ double shared[];
    double* activations = shared;
    double* derivatives = activations + hidden_size;
    double* lr_error_storage = derivatives + hidden_size;

    int tid = threadIdx.x;
    if (tid < hidden_size) {
        long long offset = static_cast<long long>(tid) * feature_count;
        double pre = static_cast<double>(hidden_biases[tid]);
        for (int f = 0; f < feature_count; ++f) {
            int8_t feature = features[f];
            if (feature == 0) {
                continue;
            }
            pre += static_cast<double>(input_weights[offset + f]) * static_cast<double>(feature);
        }
        double normalized = pre / nnue::kActivationScale;
        double tanh_val = tanh(normalized);
        activations[tid] = tanh_val * nnue::kActivationScale;
        derivatives[tid] = 1.0 - tanh_val * tanh_val;
    }
    __syncthreads();

    if (tid == 0) {
        double raw = static_cast<double>(*bias);
        for (int j = 0; j < hidden_size; ++j) {
            raw += activations[j] * static_cast<double>(output_weights[j]);
        }
        double predicted_cp = static_cast<double>(orientation) * raw * static_cast<double>(scale);
        double error = static_cast<double>(target_cp) - predicted_cp;
        double lr_error = learning_rate * error * static_cast<double>(orientation) * static_cast<double>(scale);
        lr_error_storage[0] = lr_error;

        double bias_current = static_cast<double>(*bias);
        double bias_next = bias_current + lr_error;
        if (regularisation > 0.0) {
            bias_next -= regularisation * bias_current;
        }
        *bias = clamp_weight_device(bias_next);
    }
    __syncthreads();

    if (tid >= hidden_size) {
        return;
    }

    double lr_error = lr_error_storage[0];
    double activation = activations[tid];
    double output_current = static_cast<double>(output_weights[tid]);
    double output_next = output_current + lr_error * activation;
    if (regularisation > 0.0) {
        output_next -= regularisation * output_current;
    }
    output_weights[tid] = static_cast<float>(output_next);

    double grad_pre = lr_error * output_current * derivatives[tid];
    double hidden_current = static_cast<double>(hidden_biases[tid]);
    double hidden_next = hidden_current + grad_pre;
    if (regularisation > 0.0) {
        hidden_next -= regularisation * hidden_current;
    }
    hidden_biases[tid] = clamp_weight_device(hidden_next);

    if (fabs(grad_pre) < 1e-12) {
        return;
    }

    long long offset = static_cast<long long>(tid) * feature_count;
    for (int f = 0; f < feature_count; ++f) {
        int8_t feature = features[f];
        if (feature == 0) {
            continue;
        }
        int32_t current = input_weights[offset + f];
        double next = static_cast<double>(current) + grad_pre * static_cast<double>(feature);
        if (regularisation > 0.0) {
            next -= regularisation * static_cast<double>(current);
        }
        input_weights[offset + f] = clamp_weight_device(next);
    }
}

void encode_features(const Board& board, std::vector<int8_t>& buffer) {
    std::fill(buffer.begin(), buffer.end(), 0);
    for (int color = 0; color < kNumColors; ++color) {
        for (int piece = 0; piece < kNumPieceTypes; ++piece) {
            Bitboard bb = board.pieces(static_cast<Color>(color), static_cast<PieceType>(piece));
            while (bb) {
                int square = pop_lsb(bb);
                std::size_t feature =
                    nnue::feature_index(static_cast<Color>(color), static_cast<PieceType>(piece), square);
                buffer[feature] = (color == static_cast<int>(Color::White)) ? 1 : -1;
            }
        }
    }
}

}  // namespace

void train_batch_cuda(const std::vector<TrainingExample>& batch, nnue::Network& network,
                      const Trainer::Config& config) {
    if (batch.empty()) {
        return;
    }

    int hidden = static_cast<int>(network.hidden_size());
    if (hidden <= 0) {
        return;
    }
    int feature_count = static_cast<int>(nnue::kFeatureCount);

    auto& input_weights = network.input_weights_data();
    auto& hidden_biases = network.hidden_biases_data();
    auto& output_weights = network.output_weights_data();
    int32_t bias_value = network.bias();
    float scale_value = network.scale();

    int32_t* d_input_weights = nullptr;
    int32_t* d_hidden_biases = nullptr;
    float* d_output_weights = nullptr;
    int32_t* d_bias = nullptr;
    int8_t* d_features = nullptr;

    try {
        check_cuda(cudaMalloc(&d_input_weights, input_weights.size() * sizeof(int32_t)), "cudaMalloc input weights");
        check_cuda(cudaMalloc(&d_hidden_biases, hidden_biases.size() * sizeof(int32_t)),
                   "cudaMalloc hidden biases");
        check_cuda(cudaMalloc(&d_output_weights, output_weights.size() * sizeof(float)),
                   "cudaMalloc output weights");
        check_cuda(cudaMalloc(&d_bias, sizeof(int32_t)), "cudaMalloc bias");
        check_cuda(cudaMalloc(&d_features, static_cast<size_t>(feature_count) * sizeof(int8_t)),
                   "cudaMalloc features");

        check_cuda(cudaMemcpy(d_input_weights, input_weights.data(),
                              input_weights.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy input weights to device");
        check_cuda(cudaMemcpy(d_hidden_biases, hidden_biases.data(),
                              hidden_biases.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy hidden biases to device");
        check_cuda(cudaMemcpy(d_output_weights, output_weights.data(),
                              output_weights.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy output weights to device");
        check_cuda(cudaMemcpy(d_bias, &bias_value, sizeof(int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy bias to device");

        std::vector<int8_t> feature_buffer(static_cast<std::size_t>(feature_count), 0);

        for (const TrainingExample& example : batch) {
            Board board;
            board.set_from_fen(example.fen);
            encode_features(board, feature_buffer);

            check_cuda(cudaMemcpy(d_features, feature_buffer.data(),
                                  static_cast<size_t>(feature_count) * sizeof(int8_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy features to device");

            int orientation = board.side_to_move() == Color::White ? 1 : -1;
            int target = example.target_cp;
            int threads = 1;
            while (threads < hidden) {
                threads <<= 1;
            }
            if (threads > 1024) {
                threads = 1024;
            }
            std::size_t shared_bytes = static_cast<std::size_t>(2 * hidden + 1) * sizeof(double);

            train_example_kernel<<<1, threads, shared_bytes>>>(d_features, target, orientation,
                                                               config.learning_rate, config.regularisation, hidden,
                                                               feature_count, d_input_weights, d_hidden_biases,
                                                               d_output_weights, d_bias, scale_value);
            check_cuda(cudaGetLastError(), "launch train_example_kernel");
            check_cuda(cudaDeviceSynchronize(), "train_example_kernel");
        }

        check_cuda(cudaMemcpy(input_weights.data(), d_input_weights,
                              input_weights.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy input weights to host");
        check_cuda(cudaMemcpy(hidden_biases.data(), d_hidden_biases,
                              hidden_biases.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy hidden biases to host");
        check_cuda(cudaMemcpy(output_weights.data(), d_output_weights,
                              output_weights.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy output weights to host");
        check_cuda(cudaMemcpy(&bias_value, d_bias, sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy bias to host");

        network.set_bias(bias_value);
    } catch (...) {
        if (d_features) cudaFree(d_features);
        if (d_bias) cudaFree(d_bias);
        if (d_output_weights) cudaFree(d_output_weights);
        if (d_hidden_biases) cudaFree(d_hidden_biases);
        if (d_input_weights) cudaFree(d_input_weights);
        throw;
    }

    if (d_features) cudaFree(d_features);
    if (d_bias) cudaFree(d_bias);
    if (d_output_weights) cudaFree(d_output_weights);
    if (d_hidden_biases) cudaFree(d_hidden_biases);
    if (d_input_weights) cudaFree(d_input_weights);
}

}  // namespace chiron::gpu
