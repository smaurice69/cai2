#include "tools/tuning.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <utility>

namespace chiron {

namespace {

double logistic(double elo) {
    return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
}

SelfPlayConfig prepare_config(SelfPlayConfig config) {
    config.games = 1;
    config.capture_results = false;
    config.capture_pgn = false;
    return config;
}

constexpr double kEpsilon = 1e-9;

}  // namespace

SprtTester::SprtTester(SelfPlayConfig base_config, EngineConfig baseline, EngineConfig candidate, SprtConfig sprt_config)
    : base_config_(prepare_config(std::move(base_config))),
      baseline_(std::move(baseline)),
      candidate_(std::move(candidate)),
      sprt_(std::move(sprt_config)),
      orchestrator_(std::make_unique<SelfPlayOrchestrator>(base_config_)) {
    double p0 = logistic(sprt_.elo0);
    double p1 = logistic(sprt_.elo1);
    double non_draw = std::max(1.0 - sprt_.draw_ratio, kEpsilon);
    win_prob_h0_ = std::max(p0 * non_draw, kEpsilon);
    loss_prob_h0_ = std::max((1.0 - p0) * non_draw, kEpsilon);
    win_prob_h1_ = std::max(p1 * non_draw, kEpsilon);
    loss_prob_h1_ = std::max((1.0 - p1) * non_draw, kEpsilon);
}

SprtTester::~SprtTester() = default;

double SprtTester::likelihood_increment(double candidate_score) const {
    if (candidate_score >= 1.0 - kEpsilon) {
        return std::log(win_prob_h1_ / win_prob_h0_);
    }
    if (candidate_score <= kEpsilon) {
        return std::log(loss_prob_h1_ / loss_prob_h0_);
    }
    return 0.0;  // Draws do not change the ratio in this simplified model.
}

void SprtTester::log_game(int game_index, const SelfPlayResult& result, double candidate_score,
                          std::ofstream& stream) const {
    stream << '{';
    stream << "\"game\":" << (game_index + 1) << ',';
    stream << "\"result\":\"" << result.result << "\",";
    stream << "\"termination\":\"" << result.termination << "\",";
    stream << "\"ply_count\":" << result.ply_count << ',';
    stream << "\"candidate_score\":" << candidate_score << ',';
    stream << "\"llr\":" << std::fixed << std::setprecision(5) << llr_ << std::defaultfloat;
    stream << "}\n";
}

SprtSummary SprtTester::run() {
    double upper_bound = std::log((1.0 - sprt_.beta) / sprt_.alpha);
    double lower_bound = std::log(sprt_.beta / (1.0 - sprt_.alpha));

    std::ofstream log_stream;
    if (!sprt_.results_path.empty()) {
        log_stream.open(sprt_.results_path, std::ios::out | std::ios::app);
    }

    SprtSummary summary;

    for (int game = 0; game < sprt_.max_games; ++game) {
        bool candidate_is_white = (game % 2 == 0);
        EngineConfig white = candidate_is_white ? candidate_ : baseline_;
        EngineConfig black = candidate_is_white ? baseline_ : candidate_;

        SelfPlayResult result = orchestrator_->play_game(game, white, black, false);

        double candidate_score = 0.5;
        if (result.result == "1-0") {
            candidate_score = candidate_is_white ? 1.0 : 0.0;
        } else if (result.result == "0-1") {
            candidate_score = candidate_is_white ? 0.0 : 1.0;
        }

        if (candidate_score >= 1.0 - kEpsilon) {
            ++candidate_wins_;
        } else if (candidate_score <= kEpsilon) {
            ++baseline_wins_;
        } else {
            ++draws_;
        }

        ++games_played_;
        llr_ += likelihood_increment(candidate_score);

        if (log_stream) {
            log_game(game, result, candidate_score, log_stream);
        }

        if (llr_ >= upper_bound) {
            summary.conclusion = "accept_h1";
            break;
        }
        if (llr_ <= lower_bound) {
            summary.conclusion = "accept_h0";
            break;
        }
    }

    if (summary.conclusion.empty()) {
        summary.conclusion = (games_played_ >= sprt_.max_games) ? "inconclusive" : "continue";
    }

    summary.llr = llr_;
    summary.games_played = games_played_;
    summary.candidate_wins = candidate_wins_;
    summary.baseline_wins = baseline_wins_;
    summary.draws = draws_;

    double wins = static_cast<double>(candidate_wins_) + 0.5 * static_cast<double>(draws_);
    double losses = static_cast<double>(baseline_wins_) + 0.5 * static_cast<double>(draws_);
    if (wins > 0.0 && losses > 0.0) {
        double ratio = wins / losses;
        double elo = 400.0 * std::log10(ratio);
        double variance = (1.0 / wins) + (1.0 / losses);
        double sigma = (400.0 / std::log(10.0)) * std::sqrt(variance);
        summary.elo = elo;
        summary.elo_confidence = 1.96 * sigma;
    }

    if (log_stream) {
        log_stream.flush();
    }

    return summary;
}

TimeManager::TimeManager(TimeHeuristicConfig config) : config_(config) {}

int TimeManager::allocate_time_ms(int remaining_ms, int increment_ms, int move_number, int moves_to_go) const {
    if (remaining_ms <= 0) {
        return config_.min_time_ms;
    }
    if (moves_to_go <= 0) {
        moves_to_go = 30;
    }

    double remaining = static_cast<double>(remaining_ms);
    double increment = static_cast<double>(increment_ms);
    double phase_boost = 1.0;
    if (move_number < 20) {
        phase_boost = 1.2;
    } else if (move_number > 60) {
        phase_boost = 0.8;
    }

    double allocation = remaining * config_.base_allocation * phase_boost;
    allocation += increment * config_.increment_bonus;

    double per_move_cap = remaining / static_cast<double>(moves_to_go);
    allocation = std::min(allocation, per_move_cap);

    allocation = std::clamp(allocation, static_cast<double>(config_.min_time_ms),
                             static_cast<double>(config_.max_time_ms));
    return static_cast<int>(allocation);
}

TimeTuningReport TimeManager::analyse_results_log(const std::string& path) const {
    TimeTuningReport report;
    std::ifstream stream(path);
    if (!stream) {
        return report;
    }

    long long total_ply = 0;
    std::string line;
    const std::string key = "\"ply_count\":";
    while (std::getline(stream, line)) {
        auto pos = line.find(key);
        if (pos == std::string::npos) {
            continue;
        }
        pos += key.size();
        std::istringstream iss(line.substr(pos));
        int ply = 0;
        iss >> ply;
        if (ply > 0) {
            total_ply += ply;
            ++report.games_evaluated;
        }
    }

    if (report.games_evaluated > 0) {
        report.average_ply = static_cast<double>(total_ply) / report.games_evaluated;
        report.recommended_moves_to_go = std::max(10.0, report.average_ply / 2.0);
    }

    return report;
}

}  // namespace chiron
