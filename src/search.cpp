#include "search.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>

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
    accumulator_stack_.resize(256);
    stack_.resize(256);
    killer_moves_.resize(256);
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

void Search::set_threads(int threads) { thread_count_ = std::max(1, threads); }

void Search::clear() {
    for (auto& entry : table_) {
        entry = TTEntry{};
    }
    std::memset(history_, 0, sizeof(history_));
    for (auto& killers : killer_moves_) {
        killers[0] = Move{};
        killers[1] = Move{};
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
    nodes_ = 0;
    seldepth_ = 0;
    generation_ = (generation_ + 1) & 0xFFU;

    int max_depth = std::clamp(limits.max_depth, 1, 128);
    if (static_cast<int>(accumulator_stack_.size()) < max_depth + 5) {
        accumulator_stack_.resize(static_cast<std::size_t>(max_depth) + 5);
    }
    if (static_cast<int>(stack_.size()) < max_depth + 5) {
        stack_.resize(static_cast<std::size_t>(max_depth) + 5);
    }
    if (static_cast<int>(killer_moves_.size()) < max_depth + 5) {
        killer_moves_.resize(static_cast<std::size_t>(max_depth) + 5);
    }

    repetition_stack_.clear();
    repetition_stack_.reserve(512);
    repetition_stack_.push_back(board.zobrist_key());

    evaluator_->build_accumulator(board, accumulator_stack_[0]);

    SearchResult best{};
    Move last_best{};
    int aspiration = 18;
    int previous_score = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        if (should_stop()) {
            break;
        }

        int alpha = std::max(-kInfinity, previous_score - aspiration);
        int beta = std::min(kInfinity, previous_score + aspiration);
        int score = 0;
        bool completed_window = false;

        while (true) {
            score = search_root(board, depth, alpha, beta);
            if (stop_flag.load()) {
                break;
            }
            if (score <= alpha) {
                alpha = std::max(-kInfinity, alpha - aspiration);
                aspiration *= 2;
            } else if (score >= beta) {
                beta = std::min(kInfinity, beta + aspiration);
                aspiration *= 2;
            } else {
                completed_window = true;
                break;
            }
            if (aspiration > kInfinity / 2) {
                alpha = -kInfinity;
                beta = kInfinity;
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
        best.nodes = nodes_;
        best.seldepth = seldepth_;
        best.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);
        best.pv = extract_pv(board);
        if (!best.pv.empty()) {
            best.best_move = best.pv.front();
            last_best = best.best_move;
        } else if (last_best.from != 0 || last_best.to != 0) {
            best.best_move = last_best;
        }

        if (info_callback_) {
            info_callback_(best);
        }

        if (std::abs(score) > kMateScoreThreshold) {
            break;
        }
        if (node_limit_ && nodes_ >= node_limit_) {
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

int Search::search_root(Board& board, int depth, int alpha, int beta) {
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
            lhs_score = history_score(lhs, board.side_to_move());
            rhs_score = history_score(rhs, board.side_to_move());
        }
        return lhs_score > rhs_score;
    });

    int alpha_original = alpha;
    int best_score = -kInfinity;
    Move best_move_local{};
    int move_index = 0;

    for (const Move& move : moves) {
        if (should_stop()) {
            break;
        }

        Board::State state;
        evaluator_->update_accumulator(board, move, accumulator_stack_[0], accumulator_stack_[1]);
        board.make_move(move, state);
        repetition_stack_.push_back(board.zobrist_key());

        int value = -negamax(board, depth - 1, -beta, -alpha, true, 1);

        repetition_stack_.pop_back();
        board.undo_move(move, state);

        if (stop_signal_ && stop_signal_->load()) {
            return 0;
        }

        if (value > best_score) {
            best_score = value;
            best_move_local = move;
        }

        if (value > alpha) {
            alpha = value;
        }

        ++move_index;

        if (alpha >= beta) {
            break;
        }
    }

    TTFlag flag = TTFlag::Exact;
    if (best_score <= alpha_original) {
        flag = TTFlag::Alpha;
    } else if (best_score >= beta) {
        flag = TTFlag::Beta;
    }
    store_tt(board.zobrist_key(), depth, best_score, best_move_local, static_cast<std::uint8_t>(flag), 0);
    return best_score;
}

int Search::negamax(Board& board, int depth, int alpha, int beta, bool allow_null, int ply) {
    if (should_stop()) {
        return 0;
    }

    seldepth_ = std::max(seldepth_, ply);
    ++nodes_;

    bool in_check = board.in_check(board.side_to_move());
    stack_[ply].in_check = in_check;

    if (depth <= 0) {
        return quiescence(board, alpha, beta, ply);
    }

    if (board.halfmove_clock() >= 100) {
        return 0;
    }
    if (std::count(repetition_stack_.begin(), repetition_stack_.end(), board.zobrist_key()) >= 3) {
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

    int static_eval = evaluator_->evaluate(board, accumulator_stack_[ply]);
    stack_[ply].static_eval = static_eval;
    int alpha_original = alpha;

    if (!in_check && allow_null && depth >= 3 && static_eval >= beta) {
        Board::State state;
        board.make_null_move(state);
        repetition_stack_.push_back(board.zobrist_key());
        int null_score = -negamax(board, depth - 1 - kNullMoveReduction, -beta, -beta + 1, false, ply + 1);
        repetition_stack_.pop_back();
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

    std::stable_sort(moves.begin(), moves.end(), [&](const Move& lhs, const Move& rhs) {
        if (same_move(lhs, tt_move) != same_move(rhs, tt_move)) {
            return same_move(lhs, tt_move);
        }
        if (lhs.is_capture() || rhs.is_capture()) {
            return mvv_lva(lhs, board) > mvv_lva(rhs, board);
        }
        const auto& killers = killer_moves_[ply];
        if (same_move(lhs, killers[0]) || same_move(lhs, killers[1])) {
            return true;
        }
        if (same_move(rhs, killers[0]) || same_move(rhs, killers[1])) {
            return false;
        }
        return history_score(lhs, board.side_to_move()) > history_score(rhs, board.side_to_move());
    });

    Move best_move{};
    int best_score = -kInfinity;
    int move_index = 0;

    for (const Move& move : moves) {
        Board::State state;
        evaluator_->update_accumulator(board, move, accumulator_stack_[ply], accumulator_stack_[ply + 1]);
        board.make_move(move, state);
        repetition_stack_.push_back(board.zobrist_key());

        int new_depth = depth - 1;
        bool gives_check = board.in_check(board.side_to_move());
        int score = 0;
        bool can_reduce = !move.is_capture() && !move.is_promotion() && !gives_check && !in_check && depth >= 3 && move_index >= 3;
        if (can_reduce) {
            int reduction = 1 + (move_index > 6);
            int reduced_depth = std::max(1, depth - 1 - reduction);
            score = -negamax(board, reduced_depth, -alpha - 1, -alpha, true, ply + 1);
            if (score > alpha) {
                score = -negamax(board, new_depth, -beta, -alpha, true, ply + 1);
            }
        } else {
            score = -negamax(board, new_depth, -beta, -alpha, true, ply + 1);
        }

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (score > alpha) {
            alpha = score;
        }

        repetition_stack_.pop_back();
        board.undo_move(move, state);

        if (alpha >= beta) {
            if (!move.is_capture() && !move.is_promotion()) {
                update_killers(killer_moves_[ply], move);
                update_history(move, depth, board.side_to_move());
            }
            break;
        }

        if (!move.is_capture() && !move.is_promotion() && alpha > static_eval) {
            update_history(move, depth, board.side_to_move());
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

int Search::quiescence(Board& board, int alpha, int beta, int ply) {
    if (should_stop()) {
        return 0;
    }

    ++nodes_;

    bool in_check = board.in_check(board.side_to_move());
    if (in_check) {
        return negamax(board, 1, alpha, beta, false, ply);
    }

    int stand_pat = evaluator_->evaluate(board, accumulator_stack_[ply]);
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
        evaluator_->update_accumulator(board, move, accumulator_stack_[ply], accumulator_stack_[ply + 1]);
        board.make_move(move, state);
        repetition_stack_.push_back(board.zobrist_key());
        int score = -quiescence(board, -beta, -alpha, ply + 1);
        repetition_stack_.pop_back();
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

void Search::update_history(const Move& move, int depth, Color mover) {
    if (move.is_capture() || move.is_promotion()) {
        return;
    }
    int bonus = depth * depth;
    int& entry = history_[static_cast<int>(mover)][move.from][move.to];
    entry = std::clamp(entry + bonus, -4000, 4000);
}

int Search::history_score(const Move& move, Color mover) const {
    if (move.is_capture() || move.is_promotion()) {
        return 0;
    }
    return history_[static_cast<int>(mover)][move.from][move.to];
}

bool Search::probe_tt(std::uint64_t key, int ply, TTEntry& entry) const {
    const TTEntry& probe = entry_for_key(key);
    if (probe.flag != static_cast<std::uint8_t>(TTFlag::Empty) && probe.key == key) {
        entry = probe;
        entry.score = static_cast<int16_t>(from_tt_score(entry.score, ply));
        return true;
    }
    return false;
}

void Search::store_tt(std::uint64_t key, int depth, int score, const Move& move, std::uint8_t flag, int ply) {
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
    if (node_limit_ && nodes_ >= node_limit_) {
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

}  // namespace chiron

