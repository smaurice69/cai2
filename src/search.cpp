#include "search.h"

#include <algorithm>
#include <limits>

#include "evaluation.h"

namespace chiron {

namespace {
constexpr int kMateValue = 100000;

bool same_move(const Move& a, const Move& b) {
    return a.from == b.from && a.to == b.to && a.promotion == b.promotion && a.flags == b.flags;
}

int move_order_score(const Move& move) {
    int score = 0;
    if (move.is_capture()) score += 4;
    if (move.is_promotion()) score += 2;
    if (move.is_castle()) score += 1;
    return score;
}

}  // namespace

Search::Search(std::size_t table_size, std::shared_ptr<nnue::Evaluator> evaluator)
    : evaluator_(std::move(evaluator)) {
    if (table_size == 0) {
        table_size = 1;
    }
    table_.resize(table_size);
    if (!evaluator_) {
        evaluator_ = global_evaluator();
    }
    accumulator_stack_.resize(128);
    clear();
}

void Search::clear() {
    for (auto& entry : table_) {
        entry = TTEntry{};
    }
}

Move Search::search_best_move(Board& board, int max_depth) {
    if (!evaluator_) {
        evaluator_ = global_evaluator();
    }
    evaluator_->ensure_network_loaded();
    if (accumulator_stack_.size() < static_cast<std::size_t>(max_depth + 2)) {
        accumulator_stack_.resize(static_cast<std::size_t>(max_depth + 2));
    }
    evaluator_->build_accumulator(board, accumulator_stack_[0]);

    best_move_ = Move{};
    for (int depth = 1; depth <= max_depth; ++depth) {
        alpha_beta(board, depth, -kMateValue, kMateValue, 0);
    }
    return best_move_;
}

Search::TTEntry& Search::entry_for_key(std::uint64_t key) {
    return table_[key % table_.size()];
}

const Search::TTEntry& Search::entry_for_key(std::uint64_t key) const {
    return table_[key % table_.size()];
}

bool Search::probe_tt(std::uint64_t key, TTEntry& out) const {
    const TTEntry& entry = entry_for_key(key);
    if (entry.flag != TTEntry::Flag::Empty && entry.key == key) {
        out = entry;
        return true;
    }
    return false;
}

void Search::store_tt(std::uint64_t key, int depth, int score, const Move& move, TTEntry::Flag flag) {
    TTEntry& entry = entry_for_key(key);
    if (depth >= entry.depth || flag == TTEntry::Flag::Exact) {
        entry.key = key;
        entry.depth = depth;
        entry.score = score;
        entry.move = move;
        entry.flag = flag;
    }
}

int Search::alpha_beta(Board& board, int depth, int alpha, int beta, int ply) {
    if (depth == 0) {
        return evaluator_->evaluate(board, accumulator_stack_[ply]);
    }

    TTEntry tt_entry;
    bool has_entry = probe_tt(board.zobrist_key(), tt_entry);
    if (has_entry && tt_entry.depth >= depth) {
        if (tt_entry.flag == TTEntry::Flag::Exact) {
            return tt_entry.score;
        }
        if (tt_entry.flag == TTEntry::Flag::Alpha && tt_entry.score <= alpha) {
            return tt_entry.score;
        }
        if (tt_entry.flag == TTEntry::Flag::Beta && tt_entry.score >= beta) {
            return tt_entry.score;
        }
    }

    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board);

    if (moves.empty()) {
        if (board.in_check(board.side_to_move())) {
            return -kMateValue + ply;
        }
        return 0;  // stalemate
    }

    if (has_entry) {
        auto it = std::find_if(moves.begin(), moves.end(), [&](const Move& m) { return same_move(m, tt_entry.move); });
        if (it != moves.end()) {
            std::swap(moves.front(), *it);
        }
    }

    std::stable_sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
        return move_order_score(a) > move_order_score(b);
    });

    int best_score = std::numeric_limits<int>::min();
    Move best_move_local;
    int alpha_original = alpha;

    for (const Move& move : moves) {
        Board::State state;
        evaluator_->update_accumulator(board, move, accumulator_stack_[ply], accumulator_stack_[ply + 1]);
        board.make_move(move, state);
        int score = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1);
        board.undo_move(move, state);

        if (score > best_score) {
            best_score = score;
            best_move_local = move;
            if (ply == 0) {
                best_move_ = move;
            }
        }

        if (score > alpha) {
            alpha = score;
        }

        if (alpha >= beta) {
            break;
        }
    }

    TTEntry::Flag flag = TTEntry::Flag::Exact;
    if (best_score <= alpha_original) {
        flag = TTEntry::Flag::Alpha;
    } else if (best_score >= beta) {
        flag = TTEntry::Flag::Beta;
    }

    store_tt(board.zobrist_key(), depth, best_score, best_move_local, flag);
    return best_score;
}

}  // namespace chiron

