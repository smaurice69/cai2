#include "evaluation.h"

#include <mutex>

namespace chiron {

namespace {
std::shared_ptr<nnue::Evaluator>& evaluator_instance() {
    static std::shared_ptr<nnue::Evaluator> instance = std::make_shared<nnue::Evaluator>();
    return instance;
}

std::mutex& evaluator_mutex() {
    static std::mutex mutex;
    return mutex;
}
}  // namespace

int evaluate(const Board& board) {
    auto evaluator = global_evaluator();
    nnue::Accumulator accum;
    evaluator->build_accumulator(board, accum);
    return evaluator->evaluate(board, accum);
}

std::shared_ptr<nnue::Evaluator> global_evaluator() {
    std::lock_guard<std::mutex> lock(evaluator_mutex());
    auto& instance = evaluator_instance();
    if (!instance) {
        instance = std::make_shared<nnue::Evaluator>();
    }
    instance->ensure_network_loaded();
    return instance;
}

void set_global_network_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(evaluator_mutex());
    auto& instance = evaluator_instance();
    if (!instance) {
        instance = std::make_shared<nnue::Evaluator>();
    }
    instance->set_network_path(path);
}

}  // namespace chiron

