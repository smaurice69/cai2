#pragma once

#include <string>
#include <vector>

#include "board.h"
#include "nnue/network.h"

namespace chiron {

/**
 * @brief Single training sample pairing a FEN position with a target evaluation.
 */
struct TrainingExample {
    std::string fen;   /**< Position encoded as a FEN string. */
    int target_cp = 0; /**< Target centipawn evaluation from the side to move. */
};

/**
 * @brief Lightweight wrapper managing a mutable NNUE network instance.
 */
class ParameterSet {
   public:
    ParameterSet();

    void load(const std::string& path);
    void save(const std::string& path) const;

    nnue::Network& network() { return network_; }
    const nnue::Network& network() const { return network_; }

   private:
    nnue::Network network_{};
};

/**
 * @brief Gradient-style optimiser for the simple NNUE evaluation.
 */
class Trainer {
   public:
    struct Config {
        double learning_rate = 0.05;
        double regularisation = 0.0005;
    };

    Trainer();
    explicit Trainer(Config config);

    void train_batch(const std::vector<TrainingExample>& batch, ParameterSet& parameters) const;
    int evaluate_example(const TrainingExample& example, const ParameterSet& parameters) const;

   private:
    Config config_;
};

std::vector<TrainingExample> load_training_file(const std::string& path);
void save_training_file(const std::string& path, const std::vector<TrainingExample>& data);

}  // namespace chiron

