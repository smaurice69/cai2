#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "board.h"
#include "search.h"

namespace chiron {

/**
 * @brief Minimal UCI protocol front-end driving the search engine.
 */
class UCI {
   public:
    UCI();
    ~UCI();
    void loop();

   private:
   void handle_position(const std::string& command);
   void handle_go(const std::string& command);
    void handle_setoption(const std::string& command);
    Move parse_move(const std::string& token);
    void start_search(SearchLimits limits);
    void stop_search(bool wait_for_join);
    void send_info(const SearchResult& result);
    void report_bestmove(const SearchResult& result);
    void join_thread();

    Board board_;
    Search search_;
    TimeHeuristicConfig time_config_{};
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> searching_{false};
    std::thread search_thread_;
    std::mutex io_mutex_;
    std::mutex result_mutex_;
    SearchLimits current_limits_{};
    SearchResult last_result_{};
    bool have_result_ = false;
    int move_overhead_ms_ = 30;
};

}  // namespace chiron

