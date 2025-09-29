#pragma once

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "board.h"
#include "search.h"
#include "tools/teacher.h"
#include "training/elo_tracker.h"
#include "training/trainer.h"

namespace chiron {

struct EngineConfig {
    std::string name = "Chiron";
    int max_depth = 6;
    std::size_t table_size = 1ULL << 20;
    std::string network_path;
    int threads = 1;
};

struct SelfPlayConfig {
    int games = 1;
    EngineConfig white{};
    EngineConfig black{};
    bool alternate_colors = true;
    int max_ply = 1024;
    bool capture_results = true;
    bool capture_pgn = true;
    bool record_fens = false;
    bool verbose = false;
    bool verbose_lite = false;
    std::string results_log = "selfplay_results.jsonl";
    std::string pgn_path = "selfplay_games.pgn";
    bool append_logs = true;
    unsigned int seed = 0;
    int concurrency = 1;
    bool enable_training = false;
    std::size_t training_batch_size = 256;
    double training_learning_rate = 0.05;
    std::string training_output_path = "nnue/models/chiron-selfplay-latest.nnue";
    std::string training_history_dir = "nnue/models/history";
    std::size_t training_hidden_size = nnue::kDefaultHiddenSize;
    TrainerDevice training_device = TrainerDevice::kCPU;
    bool teacher_mode = false;
    TeacherConfig teacher{};
    std::size_t teacher_chunk_size = 256;
    double randomness_temperature = 0.7;  /**< Softmax temperature for randomized move selection. */
    int randomness_max_ply = 24;          /**< Apply randomness up to this ply (0 = entire game). */
    int randomness_top_moves = 4;         /**< Consider at most this many moves when randomizing. */
    int randomness_score_margin = 40;     /**< Only randomize among moves within this score margin (cp). */
};

struct SelfPlayResult {
    std::string white_player;
    std::string black_player;
    std::string result;
    std::string termination;
    int ply_count = 0;
    std::vector<std::string> moves_san;
    std::vector<std::string> fens;
    std::string start_fen;
    std::string end_fen;
    double duration_ms = 0.0;
};

class SelfPlayOrchestrator {
   public:
    explicit SelfPlayOrchestrator(SelfPlayConfig config);

    void run();
    SelfPlayResult play_game(int game_index, const EngineConfig& white, const EngineConfig& black, bool log_outputs);

   private:
    SelfPlayResult play_single_game(int game_index, const EngineConfig& white, const EngineConfig& black);
    void log_result(int game_index, const SelfPlayResult& result);
    void write_pgn(int game_index, const SelfPlayResult& result);
    void ensure_streams();
    void handle_training(const SelfPlayResult& result);
    void log_verbose(const std::string& message);
    void log_lite(const std::string& message);
    void record_elo(int game_index, const SelfPlayResult& result);
    void log_rating_snapshot(const std::string& prefix);
    Move select_move(const SearchResult& search_result, int ply);
    void train_buffer_if_ready_locked(bool force);
    void process_teacher_batch(std::vector<std::string> fen_batch, bool force);
    void finalize_training();

    SelfPlayConfig config_;
    std::mt19937 rng_;
    std::ofstream results_stream_;
    std::ofstream pgn_stream_;
    bool streams_open_ = false;
    std::mutex log_mutex_;
    std::mutex training_mutex_;
    mutable std::mutex config_mutex_;
    std::mutex elo_mutex_;
    Trainer trainer_;
    ParameterSet parameters_;
    std::vector<TrainingExample> training_buffer_;
    std::vector<std::string> teacher_queue_;
    std::unique_ptr<TeacherEngine> teacher_engine_;
    int training_iteration_ = 0;
    std::string training_history_prefix_;
    std::string training_history_extension_;
    std::size_t total_positions_collected_ = 0;
    std::size_t total_positions_trained_ = 0;
    EloTracker elo_tracker_{};
  
    int detect_existing_history_iteration() const;
};

}  // namespace chiron

