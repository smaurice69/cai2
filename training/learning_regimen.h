#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "training/pgn_importer.h"
#include "training/selfplay.h"
#include "training/trainer.h"
#include "training/training_metrics.h"

namespace chiron {

struct LearningRegimenConfig {
    int iterations = 1;
    int selfplay_games = 8;
    int selfplay_depth = 10;
    int selfplay_concurrency = 1;
    int selfplay_max_ply = 160;
    int teacher_games = 4;
    std::string teacher_engine_path;
    int teacher_depth = 20;
    int teacher_threads = 1;
    std::string online_database_dir = "data/online_pgns";
    std::size_t online_batch_positions = 2048;
    std::size_t training_batch_size = 256;
    double learning_rate = 0.05;
    TrainerDevice training_device = TrainerDevice::kCPU;
    std::string output_network_path = "nnue/models/chiron-learned.nnue";
    std::string training_history_dir = "nnue/models/history";
    std::size_t hidden_size = nnue::kDefaultHiddenSize;
    std::size_t holdout_samples = 2048;
    bool include_draws = true;
};

class LearningRegimen {
   public:
    explicit LearningRegimen(LearningRegimenConfig config);

    void run();

   private:
    void announce_online_database_location() const;
    void ensure_directories() const;
    void run_selfplay_phase(int iteration, int total_iterations);
    void run_teacher_phase(int iteration, int total_iterations);
    void run_online_phase(int iteration, int total_iterations);
    void refresh_parameters_from_disk();
    void save_parameters();
    void log_dataset_summary(const std::string& prefix, const DatasetEvaluationResult& summary) const;
    std::vector<TrainingExample> load_online_examples(std::size_t max_positions);
    void evaluate_holdout(int iteration);

    LearningRegimenConfig config_;
    Trainer trainer_;
    ParameterSet parameters_;
    PgnImporter importer_;
    std::vector<std::filesystem::path> online_files_;
    std::size_t online_file_index_ = 0;
    std::vector<TrainingExample> holdout_set_;
    bool parameters_loaded_ = false;
    std::size_t total_positions_trained_ = 0;
};

}  // namespace chiron

