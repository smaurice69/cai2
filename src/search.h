#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "board.h"
#include "movegen.h"
#include "nnue/evaluator.h"
#include "tools/time_manager.h"

namespace chiron {

/**
 * @brief Search parameters derived from a UCI go command or self-play configuration.
 */
struct SearchLimits {
    int max_depth = 64;                    /**< Maximum iterative deepening depth. */
    std::uint64_t node_limit = 0;          /**< Optional node limit for the search. */
    int move_time_ms = -1;                 /**< Fixed time allocation for the move; overrides other timings. */
    int time_left_ms[kNumColors] = {0, 0}; /**< Remaining clock times for each color. */
    int increment_ms[kNumColors] = {0, 0}; /**< Increment gained per move for each color. */
    int moves_to_go = 0;                   /**< Moves until the next time control, if any. */
    bool infinite = false;                 /**< Search until explicitly stopped. */
    bool ponder = false;                   /**< Whether the search is in ponder mode. */
};

/**
 * @brief Aggregated information from a completed search iteration.
 */
struct SearchResult {
    Move best_move{};                                   /**< Principal variation best move. */
    int score = 0;                                      /**< Score in centipawns from the root perspective. */
    int depth = 0;                                      /**< Completed search depth. */
    int seldepth = 0;                                   /**< Maximum depth reached in the tree. */
    std::uint64_t nodes = 0;                            /**< Total nodes visited. */
    std::vector<Move> pv;                               /**< Principal variation line. */
    std::chrono::milliseconds elapsed{0};               /**< Time consumed by the search. */
};

/** Callback signature for streaming UCI info output while searching. */
using InfoCallback = std::function<void(const SearchResult&)>;

/**
 * @brief High-performance negamax searcher with modern alpha-beta enhancements.
 */
class Search {
   public:
    explicit Search(std::size_t table_size = 1ULL << 20, std::shared_ptr<nnue::Evaluator> evaluator = nullptr);

    /**
     * @brief Searches for the best move using the provided limits.
     */
    SearchResult search(Board& board, const SearchLimits& limits);

    /**
     * @brief Searches for the best move with external stop control and incremental info reporting.
     */
    SearchResult search(Board& board, const SearchLimits& limits, std::atomic<bool>& stop_flag,
                        const InfoCallback& info_cb);

    /**
     * @brief Clears the transposition table and all heuristics.
     */
    void clear();

    /**
     * @brief Reconfigures the evaluator used for static evaluations.
     */
    void set_evaluator(std::shared_ptr<nnue::Evaluator> evaluator);

    /**
     * @brief Adjusts the internal time manager heuristics.
     */
    void set_time_manager(TimeHeuristicConfig config);

    /**
     * @brief Resizes the transposition table to the requested number of entries.
     */
    void set_table_size(std::size_t entries);

    /**
     * @brief Resizes the transposition table to approximately the specified size in megabytes.
     */
    void set_table_size_mb(std::size_t megabytes);

    /**
     * @brief Configures the number of helper threads used at the root.
     */
    void set_threads(int threads);

   private:
    struct TTEntry {
        std::uint64_t key = 0ULL;
        int16_t depth = -1;
        int16_t score = 0;
        Move move{};
        std::uint8_t flag = 0;
        std::uint8_t age = 0;
    };

    struct SearchStackEntry {
        bool in_check = false;
        int static_eval = 0;
    };

    SearchResult search_impl(Board& board, const SearchLimits& limits, std::atomic<bool>& stop_flag,
                             const InfoCallback& info_cb);

    struct ThreadContext {
        std::vector<nnue::Accumulator> accumulator_stack;
        std::vector<SearchStackEntry> stack;
        std::vector<std::array<Move, 2>> killer_moves;
        int history[kNumColors][kBoardSize][kBoardSize]{};
        std::vector<std::uint64_t> repetition_stack;
    };

    int search_root(ThreadContext& ctx, Board& board, int depth, int alpha, int beta, Move& best_move);
    int search_root_worker(ThreadContext& ctx, Board& board, const Move& move, int depth, int alpha, int beta);
    int negamax(ThreadContext& ctx, Board& board, int depth, int alpha, int beta, bool allow_null, int ply);
    int quiescence(ThreadContext& ctx, Board& board, int alpha, int beta, int ply);

    void update_killers(std::array<Move, 2>& killers, const Move& move);
    void update_history(ThreadContext& ctx, const Move& move, int depth, Color mover);
    int history_score(const ThreadContext& ctx, const Move& move, Color mover) const;

    bool probe_tt(std::uint64_t key, int ply, TTEntry& entry) const;
    void store_tt(std::uint64_t key, int depth, int score, const Move& move, std::uint8_t flag, int ply);
    const TTEntry& entry_for_key(std::uint64_t key) const;
    TTEntry& entry_for_key(std::uint64_t key);

    bool should_stop() const;
    std::chrono::milliseconds compute_time_budget(const Board& board, const SearchLimits& limits) const;
    std::vector<Move> extract_pv(Board& board) const;

    void ensure_context_capacity(ThreadContext& ctx, int depth);
    void reset_context(ThreadContext& ctx);
    static void atomic_max(std::atomic<int>& target, int value);

    std::vector<TTEntry> table_;
    std::shared_ptr<nnue::Evaluator> evaluator_;
    TimeManager time_manager_{};
    std::vector<ThreadContext> contexts_;

    InfoCallback info_callback_;
    std::atomic<bool>* stop_signal_ = nullptr;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::milliseconds time_limit_{0};
    std::uint64_t node_limit_ = 0;
    std::size_t generation_ = 0;
    int thread_count_ = 1;
    mutable std::shared_mutex tt_mutex_;
    std::atomic<std::uint64_t> nodes_total_{0};
    std::atomic<int> seldepth_total_{0};
};

}  // namespace chiron

