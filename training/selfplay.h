#pragma once

#include <atomic>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "board.h"
#include "search.h"
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
    std::string results_log = "selfplay_results.jsonl";
    std::string pgn_path = "selfplay_games.pgn";
    bool append_logs = true;
    unsigned int seed = 0;
    int concurrency = 1;
    bool enable_training = false;
    std::size_t training_batch_size = 256;
    double training_learning_rate = 0.05;
    std::string training_output_path = "trained.nnue";
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

    SelfPlayConfig config_;
    std::mt19937 rng_;
    std::ofstream results_stream_;
    std::ofstream pgn_stream_;
    bool streams_open_ = false;
    std::mutex log_mutex_;
    std::mutex training_mutex_;
    Trainer trainer_;
    ParameterSet parameters_;
    std::vector<TrainingExample> training_buffer_;
};

}  // namespace chiron

