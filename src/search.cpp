#include "search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <thread>
#include <tuple>

#include "evaluation.h"

namespace chiron {

namespace {

constexpr int kInfinity = 32000;
constexpr int kMateValue = 32000;
constexpr int kMateScoreThreshold = kMateValue - 512;
constexpr int kNullMoveReduction = 2;

enum class TTFlag : std::uint8_t { Empty = 0, Exact = 1, Alpha = 2, Beta = 3 };

bool same_move(const Move& a, const Move& b) {
    return a.from == b.from && a.to == b.to && a.promotion == b.promotion && a.flags == b.flags;
}

int to_tt_score(int score, int ply) {
    if (score > kMateScoreThreshold) {
        return score + ply;
    }
    if (score < -kMateScoreThreshold) {
        return score - ply;
    }
    return score;
}

int from_tt_score(int score, int ply) {
    if (score > kMateScoreThreshold) {
        return score - ply;
    }
    if (score < -kMateScoreThreshold) {
        return score + ply;
    }
    return score;
}

int mvv_lva(const Move& move, const Board& board) {
    if (!move.is_capture()) {
        return 0;
    }
    PieceType victim = move.is_en_passant() ? PieceType::Pawn : board.piece_type_at(move.to);
    PieceType attacker = board.piece_type_at(move.from);
    static constexpr int piece_values[] = {100, 320, 330, 500, 900, 20000};
    int victim_score = piece_values[static_cast<int>(victim)];
    int attacker_score = piece_values[static_cast<int>(attacker)];
    return victim_score * 16 - attacker_score;
}

}  // namespace

Search::Search(std::size_t table_size, std::shared_ptr<nnue::Evaluator> evaluator) : evaluator_(std::move(evaluator)) {
    if (table_size == 0) {
        table_size = 1ULL;
    }
    table_.resize(table_size);
    if (!evaluator_) {
        evaluator_ = global_evaluator();
    }
    contexts_.resize(1);
    ensure_context_capacity(contexts_.front(), 128);
    reset_context(contexts_.front());
    clear();
}

void Search::set_evaluator(std::shared_ptr<nnue::Evaluator> evaluator) {
    evaluator_ = std::move(evaluator);
    if (!evaluator_) {
        evaluator_ = global_evaluator();
    }
}

void Search::set_time_manager(TimeHeuristicConfig config) { time_manager_ = TimeManager(config); }

void Search::set_table_size(std::size_t entries) {
    if (entries == 0) {
        entries = 1ULL;
    }
    std::unique_lock lock(tt_mutex_);
    table_.assign(entries, {});
    generation_ = 0;
}

void Search::set_table_size_mb(std::size_t megabytes) {
    std::size_t bytes = megabytes * 1024ULL * 1024ULL;
    std::size_t entries = bytes / sizeof(TTEntry);
    if (entries == 0) {
        entries = 1ULL;
    }
    set_table_size(entries);
}

void Search::set_threads(int threads) {
    thread_count_ = std::max(1, threads);
    contexts_.resize(static_cast<std::size_t>(thread_count_));
    for (auto& ctx : contexts_) {
        ensure_context_capacity(ctx, 128);
        reset_context(ctx);
    }
}

void Search::clear() {
    for (auto& entry : table_) {
        entry = TTEntry{};
    }
    generation_ = 0;
    for (auto& ctx : contexts_) {
        reset_context(ctx);
    }
}

SearchResult Search::search(Board& board, const SearchLimits& limits) {
    std::atomic<bool> stop_flag{false};
    return search_impl(board, limits, stop_flag, InfoCallback{});
}

SearchResult Search::search(Board& board, const SearchLimits& limits, std::atomic<bool>& stop_flag,
                            const InfoCallback& info_cb) {
    return search_impl(board, limits, stop_flag, info_cb);
}

SearchResult Search::search_impl(Board& board, const SearchLimits& limits, std::atomic<bool>& stop_flag,
                                 const InfoCallback& info_cb) {
    if (!evaluator_) {
        evaluator_ = global_evaluator();
    }
    evaluator_->ensure_network_loaded();

    info_callback_ = info_cb;
    stop_signal_ = &stop_flag;
    node_limit_ = limits.node_limit;
    start_time_ = std::chrono::steady_clock::now();
    time_limit_ = limits.infinite ? std::chrono::milliseconds::zero() : compute_time_budget(board, limits);
    nodes_total_.store(0, std::memory_order_relaxed);
    seldepth_total_.store(0, std::memory_order_relaxed);
    generation_ = (generation_ + 1) & 0xFFU;

    int max_depth = std::clamp(limits.max_depth, 1, 128);
    for (auto& ctx : contexts_) {
        ensure_context_capacity(ctx, max_depth);
        ctx.repetition_stack.clear();
        ctx.repetition_stack.reserve(512);
        std::fill(ctx.killer_moves.begin(), ctx.killer_moves.end(), std::array<Move, 2>{});
        std::memset(ctx.history, 0, sizeof(ctx.history));
    }

    ThreadContext& main_ctx = contexts_.front();
    main_ctx.repetition_stack.push_back(board.zobrist_key());

    evaluator_->build_accumulator(board, main_ctx.accumulator_stack[0]);
    for (std::size_t i = 1; i < contexts_.size(); ++i) {
        contexts_[i].accumulator_stack[0] = main_ctx.accumulator_stack[0];
        contexts_[i].repetition_stack.push_back(board.zobrist_key());
    }

    SearchResult best{};
    Move last_best{};
    int aspiration = 18;
    int previous_score = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        if (should_stop()) {
            break;
        }

        for (auto& ctx : contexts_) {
            if (ctx.repetition_stack.empty()) {
                ctx.repetition_stack.push_back(board.zobrist_key());
            } else {
                ctx.repetition_stack[0] = board.zobrist_key();
            }
            ctx.repetition_stack.resize(1);
            ctx.accumulator_stack[0] = main_ctx.accumulator_stack[0];
        }

        int alpha = std::max(-kInfinity, previous_score - aspiration);
        int beta = std::min(kInfinity, previous_score + aspiration);
        int score = 0;
        bool completed_window = false;

        Move iteration_best{};
        while (true) {
            score = search_root(main_ctx, board, depth, alpha, beta, iteration_best);
            if (stop_flag.load()) {
                break;
            }

            bool widened = false;
            if (score <= alpha) {
                if (alpha <= -kInfinity) {
                    completed_window = true;
                    break;
                }
                alpha = std::max(-kInfinity, alpha - aspiration);
                widened = true;
            } else if (score >= beta) {
                if (beta >= kInfinity) {
                    completed_window = true;
                    break;
                }
                beta = std::min(kInfinity, beta + aspiration);
                widened = true;
            } else {
                completed_window = true;
                break;
            }

            if (widened) {
                aspiration = std::min(aspiration * 2, kInfinity);
                if (aspiration > kInfinity / 2) {
                    alpha = -kInfinity;
                    beta = kInfinity;
                }
            }

            if (should_stop()) {
                break;
            }
        }

        if (stop_flag.load()) {
            break;
        }

        if (!completed_window) {
            break;
        }

        previous_score = score;
        aspiration = 18;

        best.depth = depth;
        best.score = score;
        best.nodes = nodes_total_.load(std::memory_order_relaxed);
        best.seldepth = seldepth_total_.load(std::memory_order_relaxed);
        best.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);
        best.pv = extract_pv(board);
        if (!best.pv.empty()) {
            best.best_move = best.pv.front();
            last_best = best.best_move;
        } else if (iteration_best.from != 0 || iteration_best.to != 0) {
            best.best_move = iteration_best;
            last_best = iteration_best;
        } else if (last_best.from != 0 || last_best.to != 0) {
            best.best_move = last_best;
        }

        if (info_callback_) {
            info_callback_(best);
        }

        if (std::abs(score) > kMateScoreThreshold) {
            break;
        }
        if (node_limit_ && nodes_total_.load(std::memory_order_relaxed) >= node_limit_) {
            break;
        }
    }

    if ((best.best_move.from == 0 && best.best_move.to == 0) && (last_best.from != 0 || last_best.to != 0)) {
        best.best_move = last_best;
    }

    if (best.elapsed.count() == 0) {
        best.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);
    }

    return best;
}

int Search::search_root(ThreadContext& ctx, Board& board, int depth, int alpha, int beta, Move& best_move) {
    TTEntry tt_entry;
    Move hash_move{};
    if (probe_tt(board.zobrist_key(), 0, tt_entry)) {
        hash_move = tt_entry.move;
    }

    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board);
    if (moves.empty()) {
        if (board.in_check(board.side_to_move())) {
            return -kMateValue + 1;
        }
        best_move = Move{};
        return 0;
    }

    std::stable_sort(moves.begin(), moves.end(), [&](const Move& lhs, const Move& rhs) {
        if (same_move(lhs, hash_move) != same_move(rhs, hash_move)) {
            return same_move(lhs, hash_move);
        }
        int lhs_score = 0;
        int rhs_score = 0;
        if (lhs.is_capture() || rhs.is_capture()) {
            lhs_score = mvv_lva(lhs, board);
            rhs_score = mvv_lva(rhs, board);
        } else {
            lhs_score = history_score(ctx, lhs, board.side_to_move());
            rhs_score = history_score(ctx, rhs, board.side_to_move());
        }
        return lhs_score > rhs_score;
    });

    int alpha_original = alpha;
    int best_score = -kInfinity;
    best_move = Move{};

    std::atomic<std::size_t> next_index{1};
    std::atomic<int> shared_alpha{alpha};
    std::atomic<bool> cutoff{alpha >= beta};
    std::mutex best_mutex;
    std::vector<std::thread> workers;

    // Evaluate the first move on the calling thread to seed alpha with a good bound.
    if (!moves.empty()) {
        const Move& first = moves.front();
        int value = search_root_worker(ctx, board, first, depth, alpha, beta);
        best_score = value;
        best_move = first;
        alpha = std::max(alpha, value);
        shared_alpha.store(alpha, std::memory_order_relaxed);
        if (value >= beta) {
            TTFlag flag = TTFlag::Beta;
            store_tt(board.zobrist_key(), depth, best_score, best_move, static_cast<std::uint8_t>(flag), 0);
            return best_score;
        }
    }

    auto worker_fn = [&](int context_index) {
        ThreadContext& worker_ctx = contexts_[static_cast<std::size_t>(context_index)];
        while (true) {
            if (cutoff.load(std::memory_order_relaxed) || should_stop()) {
                break;
            }
            std::size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= moves.size()) {
                break;
            }
            int local_alpha = shared_alpha.load(std::memory_order_relaxed);
            int value = search_root_worker(worker_ctx, board, moves[idx], depth, local_alpha, beta);
            if (stop_signal_ && stop_signal_->load()) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(best_mutex);
                if (value > best_score) {
                    best_score = value;
                    best_move = moves[idx];
                }
                int current_alpha = shared_alpha.load(std::memory_order_relaxed);
                if (value > current_alpha) {
                    shared_alpha.store(value, std::memory_order_relaxed);
                }
                if (value >= beta) {
                    cutoff.store(true, std::memory_order_relaxed);
                }
            }
        }
    };

    for (int i = 1; i < thread_count_; ++i) {
        workers.emplace_back(worker_fn, i);
    }

    worker_fn(0);

    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    if (best_score == -kInfinity) {
        best_score = alpha;
    }

    TTFlag flag = TTFlag::Exact;
    if (best_score <= alpha_original) {
        flag = TTFlag::Alpha;
    } else if (best_score >= beta) {
        flag = TTFlag::Beta;
    }
    store_tt(board.zobrist_key(), depth, best_score, best_move, static_cast<std::uint8_t>(flag), 0);
    return best_score;
}

int Search::search_root_worker(ThreadContext& ctx, Board& board, const Move& move, int depth, int alpha, int beta) {
    if (should_stop()) {
        return 0;
    }

    if (ctx.repetition_stack.empty()) {
        ctx.repetition_stack.push_back(board.zobrist_key());
    } else {
        ctx.repetition_stack[0] = board.zobrist_key();
    }
    ctx.repetition_stack.resize(1);

    evaluator_->update_accumulator(board, move, ctx.accumulator_stack[0], ctx.accumulator_stack[1]);

    Board local_board = board;
    Board::State state;
    local_board.make_move(move, state);
    ctx.repetition_stack.push_back(local_board.zobrist_key());

    int value = -negamax(ctx, local_board, depth - 1, -beta, -alpha, true, 1);

    ctx.repetition_stack.pop_back();
    return value;
}

int Search::negamax(ThreadContext& ctx, Board& board, int depth, int alpha, int beta, bool allow_null, int ply) {
    if (should_stop()) {
        return 0;
    }

    atomic_max(seldepth_total_, ply);
    nodes_total_.fetch_add(1, std::memory_order_relaxed);

    bool in_check = board.in_check(board.side_to_move());
    ctx.stack[ply].in_check = in_check;

    if (depth <= 0) {
        return quiescence(ctx, board, alpha, beta, ply);
    }

    if (board.halfmove_clock() >= 100) {
        return 0;
    }
    if (std::count(ctx.repetition_stack.begin(), ctx.repetition_stack.end(), board.zobrist_key()) >= 3) {
        return 0;
    }

    TTEntry tt_entry;
    Move tt_move{};
    if (probe_tt(board.zobrist_key(), ply, tt_entry)) {
        tt_move = tt_entry.move;
        if (tt_entry.depth >= depth) {
            if (tt_entry.flag == static_cast<std::uint8_t>(TTFlag::Exact)) {
                return tt_entry.score;
            }
            if (tt_entry.flag == static_cast<std::uint8_t>(TTFlag::Alpha) && tt_entry.score <= alpha) {
                return tt_entry.score;
            }
            if (tt_entry.flag == static_cast<std::uint8_t>(TTFlag::Beta) && tt_entry.score >= beta) {
                return tt_entry.score;
            }
        }
    }

    int static_eval = evaluator_->evaluate(board, ctx.accumulator_stack[ply]);
    ctx.stack[ply].static_eval = static_eval;
    int alpha_original = alpha;

    if (!in_check && allow_null && depth >= 3 && static_eval >= beta) {
        Board::State state;
        board.make_null_move(state);
        ctx.repetition_stack.push_back(board.zobrist_key());
        int null_score = -negamax(ctx, board, depth - 1 - kNullMoveReduction, -beta, -beta + 1, false, ply + 1);
        ctx.repetition_stack.pop_back();
        board.undo_null_move(state);
        if (null_score >= beta) {
            return beta;
        }
    }

    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board);
    if (moves.empty()) {
        if (in_check) {
            return -kMateValue + ply;
        }
        return 0;
    }

    const auto move_order_key = [&](const Move& move) {
        int tier = 0;
        int primary = 0;
        int secondary = 0;
        if (same_move(move, tt_move)) {
            tier = 3;
        } else if (move.is_capture()) {
            tier = 2;
            primary = mvv_lva(move, board);
        } else {
            const auto& killers = ctx.killer_moves[ply];
            if (same_move(move, killers[0])) {
                tier = 1;
                primary = 2;
            } else if (same_move(move, killers[1])) {
                tier = 1;
                primary = 1;
            }
            secondary = history_score(ctx, move, board.side_to_move());
        }
        return std::make_tuple(tier, primary, secondary);
    };

    std::stable_sort(moves.begin(), moves.end(), [&](const Move& lhs, const Move& rhs) {
        return move_order_key(lhs) > move_order_key(rhs);
    });

    Move best_move{};
    int best_score = -kInfinity;
    int move_index = 0;

    for (const Move& move : moves) {
        Board::State state;
        evaluator_->update_accumulator(board, move, ctx.accumulator_stack[ply], ctx.accumulator_stack[ply + 1]);
        board.make_move(move, state);
        ctx.repetition_stack.push_back(board.zobrist_key());

        int new_depth = depth - 1;
        bool gives_check = board.in_check(board.side_to_move());
        int score = 0;
        bool can_reduce = !move.is_capture() && !move.is_promotion() && !gives_check && !in_check && depth >= 3 && move_index >= 3;
        if (can_reduce) {
            int reduction = 1 + (move_index > 6);
            int reduced_depth = std::max(1, depth - 1 - reduction);
            score = -negamax(ctx, board, reduced_depth, -alpha - 1, -alpha, true, ply + 1);
            if (score > alpha) {
                score = -negamax(ctx, board, new_depth, -beta, -alpha, true, ply + 1);
            }
        } else {
            score = -negamax(ctx, board, new_depth, -beta, -alpha, true, ply + 1);
        }

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (score > alpha) {
            alpha = score;
        }

        ctx.repetition_stack.pop_back();
        board.undo_move(move, state);

        if (alpha >= beta) {
            if (!move.is_capture() && !move.is_promotion()) {
                update_killers(ctx.killer_moves[ply], move);
                update_history(ctx, move, depth, board.side_to_move());
            }
            break;
        }

        if (!move.is_capture() && !move.is_promotion() && alpha > static_eval) {
            update_history(ctx, move, depth, board.side_to_move());
        }

        ++move_index;
    }

    if (best_move.from == 0 && best_move.to == 0) {
        best_move = moves.front();
    }

    TTFlag flag = TTFlag::Exact;
    if (best_score <= alpha_original) {
        flag = TTFlag::Alpha;
    } else if (best_score >= beta) {
        flag = TTFlag::Beta;
    }
    store_tt(board.zobrist_key(), depth, best_score, best_move, static_cast<std::uint8_t>(flag), ply);
    return best_score;
}

int Search::quiescence(ThreadContext& ctx, Board& board, int alpha, int beta, int ply) {
    if (should_stop()) {
        return 0;
    }

    nodes_total_.fetch_add(1, std::memory_order_relaxed);

    bool in_check = board.in_check(board.side_to_move());
    if (in_check) {
        return negamax(ctx, board, 1, alpha, beta, false, ply);
    }

    int stand_pat = evaluator_->evaluate(board, ctx.accumulator_stack[ply]);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board);
    std::vector<Move> captures;
    captures.reserve(moves.size());
    for (const Move& move : moves) {
        if (move.is_capture() || move.is_promotion()) {
            captures.push_back(move);
        }
    }

    std::sort(captures.begin(), captures.end(), [&](const Move& lhs, const Move& rhs) {
        return mvv_lva(lhs, board) > mvv_lva(rhs, board);
    });

    for (const Move& move : captures) {
        Board::State state;
        evaluator_->update_accumulator(board, move, ctx.accumulator_stack[ply], ctx.accumulator_stack[ply + 1]);
        board.make_move(move, state);
        ctx.repetition_stack.push_back(board.zobrist_key());
        int score = -quiescence(ctx, board, -beta, -alpha, ply + 1);
        ctx.repetition_stack.pop_back();
        board.undo_move(move, state);

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

void Search::update_killers(std::array<Move, 2>& killers, const Move& move) {
    if (same_move(move, killers[0])) {
        return;
    }
    killers[1] = killers[0];
    killers[0] = move;
}

void Search::update_history(ThreadContext& ctx, const Move& move, int depth, Color mover) {
    if (move.is_capture() || move.is_promotion()) {
        return;
    }
    int bonus = depth * depth;
    int& entry = ctx.history[static_cast<int>(mover)][move.from][move.to];
    entry = std::clamp(entry + bonus, -4000, 4000);
}

int Search::history_score(const ThreadContext& ctx, const Move& move, Color mover) const {
    if (move.is_capture() || move.is_promotion()) {
        return 0;
    }
    return ctx.history[static_cast<int>(mover)][move.from][move.to];
}

bool Search::probe_tt(std::uint64_t key, int ply, TTEntry& entry) const {
    std::shared_lock lock(tt_mutex_);
    const TTEntry& probe = entry_for_key(key);
    if (probe.flag != static_cast<std::uint8_t>(TTFlag::Empty) && probe.key == key) {
        entry = probe;
        entry.score = static_cast<int16_t>(from_tt_score(entry.score, ply));
        return true;
    }
    return false;
}

void Search::store_tt(std::uint64_t key, int depth, int score, const Move& move, std::uint8_t flag, int ply) {
    std::unique_lock lock(tt_mutex_);
    TTEntry& entry = entry_for_key(key);
    int stored = to_tt_score(score, ply);
    if (entry.flag == static_cast<std::uint8_t>(TTFlag::Empty) || entry.depth <= depth || entry.age != generation_) {
        entry.key = key;
        entry.depth = static_cast<int16_t>(depth);
        entry.score = static_cast<int16_t>(stored);
        entry.move = move;
        entry.flag = flag;
        entry.age = static_cast<std::uint8_t>(generation_);
    }
}

const Search::TTEntry& Search::entry_for_key(std::uint64_t key) const {
    return table_[key % table_.size()];
}

Search::TTEntry& Search::entry_for_key(std::uint64_t key) { return table_[key % table_.size()]; }

bool Search::should_stop() const {
    if (stop_signal_ && stop_signal_->load()) {
        return true;
    }
    if (node_limit_ && nodes_total_.load(std::memory_order_relaxed) >= node_limit_) {
        return true;
    }
    if (time_limit_.count() > 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);
        if (elapsed >= time_limit_) {
            return true;
        }
    }
    return false;
}

std::chrono::milliseconds Search::compute_time_budget(const Board& board, const SearchLimits& limits) const {
    if (limits.move_time_ms >= 0) {
        return std::chrono::milliseconds(limits.move_time_ms);
    }
    if (limits.infinite) {
        return std::chrono::milliseconds(0);
    }
    Color us = board.side_to_move();
    int time_left = limits.time_left_ms[static_cast<int>(us)];
    int increment = limits.increment_ms[static_cast<int>(us)];
    if (time_left <= 0 && increment <= 0) {
        return std::chrono::milliseconds(0);
    }
    int move_number = board.fullmove_number();
    int allocation = time_manager_.allocate_time_ms(time_left, increment, move_number, limits.moves_to_go);
    return std::chrono::milliseconds(std::max(allocation, 0));
}

std::vector<Move> Search::extract_pv(Board& board) const {
    std::vector<Move> pv;
    Board copy = board;
    std::vector<Board::State> states;
    states.reserve(64);
    for (int depth = 0; depth < 64; ++depth) {
        std::uint64_t key = copy.zobrist_key();
        std::shared_lock lock(tt_mutex_);
        const TTEntry& entry = entry_for_key(key);
        if (entry.flag == static_cast<std::uint8_t>(TTFlag::Empty) || entry.key != key) {
            break;
        }
        Move move = entry.move;
        if (move.from == move.to && move.from == 0) {
            break;
        }
        pv.push_back(move);
        Board::State state;
        copy.make_move(move, state);
        states.push_back(state);
        if (MoveGenerator::generate_legal_moves(copy).empty()) {
            break;
        }
    }
    return pv;
}

void Search::ensure_context_capacity(ThreadContext& ctx, int depth) {
    int required = std::max(depth + 5, 64);
    if (static_cast<int>(ctx.accumulator_stack.size()) < required) {
        ctx.accumulator_stack.resize(static_cast<std::size_t>(required));
    }
    if (static_cast<int>(ctx.stack.size()) < required) {
        ctx.stack.resize(static_cast<std::size_t>(required));
    }
    if (static_cast<int>(ctx.killer_moves.size()) < required) {
        std::size_t old_size = ctx.killer_moves.size();
        ctx.killer_moves.resize(static_cast<std::size_t>(required));
        for (std::size_t i = old_size; i < ctx.killer_moves.size(); ++i) {
            ctx.killer_moves[i] = {Move{}, Move{}};
        }
    }
}

void Search::reset_context(ThreadContext& ctx) {
    for (auto& killers : ctx.killer_moves) {
        killers[0] = Move{};
        killers[1] = Move{};
    }
    std::memset(ctx.history, 0, sizeof(ctx.history));
    ctx.repetition_stack.clear();
}

void Search::atomic_max(std::atomic<int>& target, int value) {
    int current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

}  // namespace chiron

