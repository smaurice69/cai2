#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace chiron {

/**
 * @brief Lightweight Elo rating accumulator for tracking self-play progress.
 */
class EloTracker {
   public:
    struct PlayerSummary {
        std::string name;
        double rating = 0.0;
        double delta = 0.0;
        int games = 0;
        int wins = 0;
        int draws = 0;
        int losses = 0;
        double score = 0.0;
    };

    struct GameUpdate {
        PlayerSummary white;
        PlayerSummary black;
        double expected_white = 0.5;
        double result = 0.5;
    };

    EloTracker(double initial_rating = 1500.0, double k_factor = 24.0);

    /**
     * @brief Record a completed game and update both players' ratings.
     * @param white Name of the white player.
     * @param black Name of the black player.
     * @param white_score Result for white (1 = win, 0.5 = draw, 0 = loss).
     * @return Rating update details for both players.
     */
    GameUpdate record_game(const std::string& white, const std::string& black, double white_score);

    /**
     * @brief Snapshot of all tracked players sorted by rating.
     */
    std::vector<PlayerSummary> snapshot() const;

   private:
    struct InternalStats {
        double rating = 0.0;
        int games = 0;
        int wins = 0;
        int draws = 0;
        int losses = 0;
        double score = 0.0;
    };

    double initial_rating_;
    double k_factor_;
    std::unordered_map<std::string, InternalStats> players_;
};

}  // namespace chiron

