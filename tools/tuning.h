#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "tools/time_manager.h"
#include "training/selfplay.h"

namespace chiron {

struct SprtConfig {
    double alpha = 0.05;
    double beta = 0.05;
    double elo0 = 0.0;
    double elo1 = 10.0;
    double draw_ratio = 0.5;
    int max_games = 200;
    std::string results_path = "sprt_results.jsonl";
};

struct SprtSummary {
    std::string conclusion;
    double llr = 0.0;
    int games_played = 0;
    int candidate_wins = 0;
    int baseline_wins = 0;
    int draws = 0;
};

class SprtTester {
   public:
    SprtTester(SelfPlayConfig base_config, EngineConfig baseline, EngineConfig candidate, SprtConfig sprt_config);
    ~SprtTester();

    SprtSummary run();

   private:
    double likelihood_increment(double candidate_score) const;
    void log_game(int game_index, const SelfPlayResult& result, double candidate_score, std::ofstream& stream) const;

    SelfPlayConfig base_config_;
    EngineConfig baseline_;
    EngineConfig candidate_;
    SprtConfig sprt_;
    std::unique_ptr<SelfPlayOrchestrator> orchestrator_;

    double llr_ = 0.0;
    int games_played_ = 0;
    int candidate_wins_ = 0;
    int baseline_wins_ = 0;
    int draws_ = 0;
    double win_prob_h0_ = 0.0;
    double win_prob_h1_ = 0.0;
    double loss_prob_h0_ = 0.0;
    double loss_prob_h1_ = 0.0;
};

}  // namespace chiron

