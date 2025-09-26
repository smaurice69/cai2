#include "training/elo_tracker.h"

#include <algorithm>
#include <cmath>

namespace chiron {

namespace {
double expected_score(double rating_a, double rating_b) {
    return 1.0 / (1.0 + std::pow(10.0, (rating_b - rating_a) / 400.0));
}

}  // namespace

EloTracker::EloTracker(double initial_rating, double k_factor)
    : initial_rating_(initial_rating), k_factor_(k_factor) {}

EloTracker::GameUpdate EloTracker::record_game(const std::string& white, const std::string& black, double white_score) {
    auto [white_it, white_inserted] = players_.try_emplace(white);
    auto [black_it, black_inserted] = players_.try_emplace(black);
    if (white_inserted) {
        white_it->second.rating = initial_rating_;
    }
    if (black_inserted) {
        black_it->second.rating = initial_rating_;
    }

    InternalStats& white_stats = white_it->second;
    InternalStats& black_stats = black_it->second;

    double expected_white = expected_score(white_stats.rating, black_stats.rating);
    double expected_black = 1.0 - expected_white;

    double previous_white = white_stats.rating;
    double previous_black = black_stats.rating;

    double black_score = 1.0 - white_score;
    white_stats.rating += k_factor_ * (white_score - expected_white);
    black_stats.rating += k_factor_ * (black_score - expected_black);

    if (white_score > 0.75) {
        ++white_stats.wins;
        ++black_stats.losses;
    } else if (white_score < 0.25) {
        ++white_stats.losses;
        ++black_stats.wins;
    } else {
        ++white_stats.draws;
        ++black_stats.draws;
    }

    ++white_stats.games;
    ++black_stats.games;
    white_stats.score += white_score;
    black_stats.score += black_score;

    GameUpdate update;
    update.white.name = white;
    update.white.rating = white_stats.rating;
    update.white.delta = white_stats.rating - previous_white;
    update.white.games = white_stats.games;
    update.white.wins = white_stats.wins;
    update.white.draws = white_stats.draws;
    update.white.losses = white_stats.losses;
    update.white.score = white_stats.score;

    update.black.name = black;
    update.black.rating = black_stats.rating;
    update.black.delta = black_stats.rating - previous_black;
    update.black.games = black_stats.games;
    update.black.wins = black_stats.wins;
    update.black.draws = black_stats.draws;
    update.black.losses = black_stats.losses;
    update.black.score = black_stats.score;

    update.expected_white = expected_white;
    update.result = white_score;
    return update;
}

std::vector<EloTracker::PlayerSummary> EloTracker::snapshot() const {
    std::vector<PlayerSummary> table;
    table.reserve(players_.size());
    for (const auto& [name, stats] : players_) {
        PlayerSummary summary;
        summary.name = name;
        summary.rating = stats.rating == 0.0 && stats.games == 0 ? initial_rating_ : stats.rating;
        summary.games = stats.games;
        summary.wins = stats.wins;
        summary.draws = stats.draws;
        summary.losses = stats.losses;
        summary.score = stats.score;
        table.push_back(summary);
    }
    std::sort(table.begin(), table.end(), [](const PlayerSummary& lhs, const PlayerSummary& rhs) {
        if (std::fabs(lhs.rating - rhs.rating) > 1e-6) {
            return lhs.rating > rhs.rating;
        }
        return lhs.name < rhs.name;
    });
    return table;
}

}  // namespace chiron

