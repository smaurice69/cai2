#include <gtest/gtest.h>

#include "training/selfplay.h"

namespace chiron {

TEST(SelfPlay, GeneratesGameData) {
    SelfPlayConfig config;
    config.games = 1;
    config.white.max_depth = 1;
    config.black.max_depth = 1;
    config.capture_results = false;
    config.capture_pgn = false;
    config.max_ply = 40;

    SelfPlayOrchestrator orchestrator(config);
    SelfPlayResult result = orchestrator.play_game(0, config.white, config.black, false);
    EXPECT_GE(result.ply_count, 0);
    EXPECT_FALSE(result.result.empty());
}

}  // namespace chiron

