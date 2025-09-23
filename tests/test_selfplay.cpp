#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

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

TEST(SelfPlay, LogsWellFormedResultLine) {
    namespace fs = std::filesystem;

    auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temp_file = fs::temp_directory_path() /
                        fs::path("selfplay-log-" + std::to_string(timestamp) + ".jsonl");

    SelfPlayConfig config;
    config.games = 1;
    config.white.max_depth = 1;
    config.black.max_depth = 1;
    config.capture_results = true;
    config.capture_pgn = false;
    config.append_logs = false;
    config.results_log = temp_file.string();
    config.max_ply = 40;

    SelfPlayOrchestrator orchestrator(config);
    orchestrator.play_game(0, config.white, config.black, true);

    std::ifstream log_stream(temp_file);
    ASSERT_TRUE(log_stream.is_open());

    std::string line;
    std::getline(log_stream, line);
    log_stream.close();
    fs::remove(temp_file);

    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.front(), '{');
    EXPECT_EQ(line.back(), '}');

    auto white_pos = line.find("\"white\":\"");
    ASSERT_NE(white_pos, std::string::npos);
    EXPECT_EQ(line.find("\"white\":\"", white_pos + 1), std::string::npos);

    EXPECT_EQ(line.find(",\"\""), std::string::npos);
}

}  // namespace chiron

