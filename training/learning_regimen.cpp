#include "training/learning_regimen.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace chiron {

namespace {

bool has_pgn_extension(const std::filesystem::path& path) {
    if (!path.has_extension()) {
        return false;
    }
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".pgn";
}

std::string timestamp_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

}  // namespace

LearningRegimen::LearningRegimen(LearningRegimenConfig config)
    : config_(std::move(config)),
      trainer_(Trainer::Config{config_.learning_rate, 0.0005, config_.training_device}),
      parameters_(config_.hidden_size) {
    ensure_directories();

    if (!config_.output_network_path.empty() && std::filesystem::exists(config_.output_network_path)) {
        parameters_.load(config_.output_network_path);
        config_.hidden_size = parameters_.network().hidden_size();
        parameters_loaded_ = true;
    }

    if (!config_.online_database_dir.empty()) {
        std::filesystem::path dir(config_.online_database_dir);
        if (std::filesystem::exists(dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.is_regular_file() && has_pgn_extension(entry.path())) {
                    online_files_.push_back(entry.path());
                }
            }
        }
        std::sort(online_files_.begin(), online_files_.end());
    }

    if (!online_files_.empty() && config_.holdout_samples > 0) {
        std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
        for (const auto& path : online_files_) {
            try {
                std::vector<TrainingExample> examples = importer_.import_file(path.string(), config_.include_draws);
                if (examples.empty()) {
                    continue;
                }
                std::shuffle(examples.begin(), examples.end(), rng);
                for (const TrainingExample& example : examples) {
                    holdout_set_.push_back(example);
                    if (holdout_set_.size() >= config_.holdout_samples) {
                        break;
                    }
                }
            } catch (const std::exception&) {
                continue;
            }
            if (holdout_set_.size() >= config_.holdout_samples) {
                break;
            }
        }
        if (holdout_set_.size() > config_.holdout_samples) {
            holdout_set_.resize(config_.holdout_samples);
        }
    }
}

void LearningRegimen::ensure_directories() const {
    if (!config_.output_network_path.empty()) {
        std::filesystem::path output_path(config_.output_network_path);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
    }
    if (!config_.training_history_dir.empty()) {
        std::filesystem::create_directories(config_.training_history_dir);
    }
    if (!config_.online_database_dir.empty()) {
        std::filesystem::create_directories(config_.online_database_dir);
    }
}

void LearningRegimen::announce_online_database_location() const {
    std::cout << "[Learn] Online database directory: " << config_.online_database_dir << '\n';
    if (online_files_.empty()) {
        std::cout << "[Learn] Place raw PGN files from online sources into this directory. They will be parsed on the fly.\n";
    } else {
        std::cout << "[Learn] Found " << online_files_.size()
                  << " PGN file(s). They will be cycled through during training." << std::endl;
    }
}

void LearningRegimen::refresh_parameters_from_disk() {
    if (!parameters_loaded_) {
        parameters_.reset(config_.hidden_size);
        parameters_loaded_ = true;
    }
    if (!config_.output_network_path.empty() && std::filesystem::exists(config_.output_network_path)) {
        parameters_.load(config_.output_network_path);
        config_.hidden_size = parameters_.network().hidden_size();
    }
}

void LearningRegimen::save_parameters() {
    if (config_.output_network_path.empty()) {
        return;
    }
    parameters_.save(config_.output_network_path);
}

void LearningRegimen::log_dataset_summary(const std::string& prefix, const DatasetEvaluationResult& summary) const {
    if (summary.samples == 0) {
        return;
    }
    std::cout << prefix << std::fixed << std::setprecision(1) << summary.pseudo_elo
              << ", accuracy " << std::setprecision(1) << (summary.accuracy * 100.0)
              << "% over " << summary.samples << " samples" << std::defaultfloat << std::setprecision(6)
              << std::endl;
}

std::vector<TrainingExample> LearningRegimen::load_online_examples(std::size_t max_positions) {
    std::vector<TrainingExample> result;
    if (online_files_.empty() || max_positions == 0) {
        return result;
    }

    std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
    std::size_t attempts = 0;
    while (result.size() < max_positions && attempts < online_files_.size()) {
        const std::filesystem::path& path = online_files_[online_file_index_];
        online_file_index_ = (online_file_index_ + 1) % online_files_.size();
        ++attempts;
        try {
            std::vector<TrainingExample> examples = importer_.import_file(path.string(), config_.include_draws);
            if (examples.empty()) {
                continue;
            }
            std::shuffle(examples.begin(), examples.end(), rng);
            std::size_t needed = max_positions - result.size();
            if (examples.size() > needed) {
                examples.resize(needed);
            }
            result.insert(result.end(), examples.begin(), examples.end());
        } catch (const std::exception& ex) {
            std::cout << "[Learn] Warning: failed to read " << path << ": " << ex.what() << std::endl;
        }
    }
    return result;
}

void LearningRegimen::run_selfplay_phase(int iteration, int total_iterations) {
    if (config_.selfplay_games <= 0) {
        return;
    }
    std::cout << "[Learn] Iteration " << iteration << '/' << total_iterations << " self-play: "
              << config_.selfplay_games << " games (depth " << config_.selfplay_depth << ")" << std::endl;

    SelfPlayConfig sp;
    sp.games = config_.selfplay_games;
    sp.max_ply = config_.selfplay_max_ply;
    sp.concurrency = std::max(1, config_.selfplay_concurrency);
    sp.enable_training = true;
    sp.training_batch_size = config_.training_batch_size;
    sp.training_learning_rate = config_.learning_rate;
    sp.training_device = config_.training_device;
    sp.training_output_path = config_.output_network_path;
    sp.training_history_dir = config_.training_history_dir;
    sp.training_hidden_size = config_.hidden_size;
    sp.white.max_depth = config_.selfplay_depth;
    sp.black.max_depth = config_.selfplay_depth;
    sp.white.name = "Chiron";
    sp.black.name = "Chiron";
    sp.capture_results = false;
    sp.capture_pgn = false;
    sp.verbose_lite = true;
    sp.teacher_mode = false;

    SelfPlayOrchestrator orchestrator(sp);
    orchestrator.run();
    refresh_parameters_from_disk();
}

void LearningRegimen::run_teacher_phase(int iteration, int total_iterations) {
    if (config_.teacher_games <= 0 || config_.teacher_engine_path.empty()) {
        return;
    }
    std::cout << "[Learn] Iteration " << iteration << '/' << total_iterations << " teacher-guided self-play: "
              << config_.teacher_games << " games using " << config_.teacher_engine_path << std::endl;

    SelfPlayConfig teacher_sp;
    teacher_sp.games = config_.teacher_games;
    teacher_sp.max_ply = config_.selfplay_max_ply;
    teacher_sp.concurrency = std::max(1, config_.selfplay_concurrency);
    teacher_sp.enable_training = true;
    teacher_sp.training_batch_size = config_.training_batch_size;
    teacher_sp.training_learning_rate = config_.learning_rate;
    teacher_sp.training_device = config_.training_device;
    teacher_sp.training_output_path = config_.output_network_path;
    teacher_sp.training_history_dir = config_.training_history_dir;
    teacher_sp.training_hidden_size = config_.hidden_size;
    teacher_sp.white.max_depth = config_.selfplay_depth;
    teacher_sp.black.max_depth = config_.selfplay_depth;
    teacher_sp.capture_results = false;
    teacher_sp.capture_pgn = false;
    teacher_sp.verbose_lite = true;
    teacher_sp.teacher_mode = true;
    teacher_sp.teacher.engine_path = config_.teacher_engine_path;
    teacher_sp.teacher.depth = config_.teacher_depth;
    teacher_sp.teacher.threads = config_.teacher_threads;
    teacher_sp.teacher_chunk_size = config_.training_batch_size;

    SelfPlayOrchestrator orchestrator(teacher_sp);
    orchestrator.run();
    refresh_parameters_from_disk();
}

void LearningRegimen::run_online_phase(int iteration, int total_iterations) {
    if (config_.online_batch_positions == 0) {
        return;
    }
    std::vector<TrainingExample> dataset = load_online_examples(config_.online_batch_positions);
    if (dataset.empty()) {
        std::cout << "[Learn] Iteration " << iteration << '/' << total_iterations
                  << " online phase skipped (no PGN data available)." << std::endl;
        return;
    }

    std::cout << "[Learn] Iteration " << iteration << '/' << total_iterations << " online replay: "
              << dataset.size() << " positions from PGNs" << std::endl;

    refresh_parameters_from_disk();

    std::size_t batch_size = std::max<std::size_t>(1, config_.training_batch_size);
    for (std::size_t offset = 0; offset < dataset.size(); offset += batch_size) {
        std::size_t end = std::min(offset + batch_size, dataset.size());
        std::vector<TrainingExample> batch(dataset.begin() + static_cast<std::ptrdiff_t>(offset),
                                           dataset.begin() + static_cast<std::ptrdiff_t>(end));
        trainer_.train_batch(batch, parameters_);
    }
    total_positions_trained_ += dataset.size();
    save_parameters();

    DatasetEvaluationResult summary =
        evaluate_dataset_performance(dataset, parameters_, trainer_, std::min<std::size_t>(dataset.size(), 4096));
    log_dataset_summary("[Learn] Online replay pseudo-Elo ", summary);
}

void LearningRegimen::evaluate_holdout(int iteration) {
    if (holdout_set_.empty()) {
        return;
    }
    refresh_parameters_from_disk();
    DatasetEvaluationResult summary = evaluate_dataset_performance(holdout_set_, parameters_, trainer_,
                                                                   std::min(config_.holdout_samples, holdout_set_.size()));
    std::ostringstream prefix;
    prefix << "[Learn] Holdout after iteration " << iteration << " pseudo-Elo ";
    log_dataset_summary(prefix.str(), summary);
}

void LearningRegimen::run() {
    announce_online_database_location();
    if (!holdout_set_.empty()) {
        std::cout << "[Learn] Using " << holdout_set_.size() << " holdout samples for progress tracking." << std::endl;
    }

    for (int iteration = 1; iteration <= config_.iterations; ++iteration) {
        std::cout << "[Learn] === Iteration " << iteration << " started at " << timestamp_string() << " ===" << std::endl;
        run_selfplay_phase(iteration, config_.iterations);
        run_teacher_phase(iteration, config_.iterations);
        run_online_phase(iteration, config_.iterations);
        evaluate_holdout(iteration);
        std::cout << "[Learn] Iteration " << iteration << " complete. Cumulative supervised samples: "
                  << total_positions_trained_ << std::endl;
    }

    std::cout << "[Learn] Training complete. Latest network saved to " << config_.output_network_path << std::endl;
}

}  // namespace chiron

