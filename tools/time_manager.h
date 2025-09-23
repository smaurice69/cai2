#pragma once

#include <string>

namespace chiron {

struct TimeHeuristicConfig {
    double base_allocation = 0.04;  // Fraction of remaining time to invest each move.
    double increment_bonus = 0.5;   // Additional fraction of increment to invest.
    int min_time_ms = 10;
    int max_time_ms = 2000;
};

struct TimeTuningReport {
    int games_evaluated = 0;
    double average_ply = 0.0;
    double recommended_moves_to_go = 40.0;
};

class TimeManager {
   public:
    explicit TimeManager(TimeHeuristicConfig config = {});

    int allocate_time_ms(int remaining_ms, int increment_ms, int move_number, int moves_to_go) const;
    TimeTuningReport analyse_results_log(const std::string& path) const;

   private:
    TimeHeuristicConfig config_;
};

}  // namespace chiron

