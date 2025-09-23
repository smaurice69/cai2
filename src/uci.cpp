#include "uci.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluation.h"

namespace chiron {

namespace {
constexpr int kMateValue = 32000;
constexpr int kMateThreshold = kMateValue - 512;

std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

}  // namespace

UCI::UCI() : board_(), search_(1 << 20) {
    board_.set_start_position();
    search_.set_time_manager(time_config_);
    search_.set_table_size_mb(16);
}

UCI::~UCI() { stop_search(true); }

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") {
            std::cout << "id name Chiron" << std::endl;
            std::cout << "id author OpenAI Assistant" << std::endl;
            std::cout << "option name Hash type spin default 16 min 1 max 4096" << std::endl;
            std::cout << "option name Threads type spin default 1 min 1 max 128" << std::endl;
            std::cout << "option name Move Overhead type spin default " << move_overhead_ms_
                      << " min 0 max 5000" << std::endl;
            std::cout << "option name Base Time Percent type spin default "
                      << static_cast<int>(time_config_.base_allocation * 100.0) << " min 1 max 100" << std::endl;
            std::cout << "option name Increment Percent type spin default "
                      << static_cast<int>(time_config_.increment_bonus * 100.0) << " min 0 max 500" << std::endl;
            std::cout << "option name Minimum Think Time type spin default " << time_config_.min_time_ms
                      << " min 1 max 10000" << std::endl;
            std::cout << "option name Maximum Think Time type spin default " << time_config_.max_time_ms
                      << " min 10 max 120000" << std::endl;
            std::cout << "option name EvalNetwork type string default " << std::endl;
            std::cout << "option name Ponder type check default false" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (line == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (line == "ucinewgame") {
            stop_search(true);
            board_.set_start_position();
            search_.clear();
        } else if (line.rfind("setoption", 0) == 0) {
            handle_setoption(line);
        } else if (line.rfind("position", 0) == 0) {
            stop_search(true);
            handle_position(line);
        } else if (line.rfind("go", 0) == 0) {
            handle_go(line);
        } else if (line == "stop") {
            stop_search(true);
        } else if (line == "quit") {
            stop_search(true);
            break;
        }
    }
}

void UCI::handle_position(const std::string& command) {
    auto tokens = tokenize(command);
    if (tokens.size() < 2) {
        return;
    }

    std::size_t index = 1;
    if (tokens[index] == "startpos") {
        board_.set_start_position();
        ++index;
    } else if (tokens[index] == "fen") {
        if (tokens.size() < index + 7) {
            throw std::runtime_error("Incomplete FEN in position command");
        }
        std::ostringstream fen;
        for (int i = 0; i < 6; ++i) {
            if (i > 0) {
                fen << ' ';
            }
            fen << tokens[index + 1 + i];
        }
        board_.set_from_fen(fen.str());
        index += 7;
    }

    if (index < tokens.size() && tokens[index] == "moves") {
        ++index;
        while (index < tokens.size()) {
            Move move = parse_move(tokens[index]);
            Board::State state;
            board_.make_move(move, state);
            ++index;
        }
    }
}

void UCI::handle_go(const std::string& command) {
    SearchLimits limits;
    limits.max_depth = 64;
    auto tokens = tokenize(command);
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        auto next_int = [&](int& destination) {
            if (i + 1 >= tokens.size()) {
                return;
            }
            destination = std::stoi(tokens[++i]);
        };
        auto next_uint64 = [&](std::uint64_t& destination) {
            if (i + 1 >= tokens.size()) {
                return;
            }
            destination = static_cast<std::uint64_t>(std::stoll(tokens[++i]));
        };
        if (token == "wtime") {
            next_int(limits.time_left_ms[static_cast<int>(Color::White)]);
        } else if (token == "btime") {
            next_int(limits.time_left_ms[static_cast<int>(Color::Black)]);
        } else if (token == "winc") {
            next_int(limits.increment_ms[static_cast<int>(Color::White)]);
        } else if (token == "binc") {
            next_int(limits.increment_ms[static_cast<int>(Color::Black)]);
        } else if (token == "movestogo") {
            next_int(limits.moves_to_go);
        } else if (token == "depth") {
            next_int(limits.max_depth);
        } else if (token == "nodes") {
            next_uint64(limits.node_limit);
        } else if (token == "movetime") {
            next_int(limits.move_time_ms);
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        } else if (token == "mate") {
            int mate_depth = 0;
            next_int(mate_depth);
            if (mate_depth > 0) {
                limits.max_depth = mate_depth * 2;
            }
        }
    }

    if (limits.max_depth <= 0) {
        limits.max_depth = 64;
    }

    start_search(limits);
}

void UCI::handle_setoption(const std::string& command) {
    std::istringstream iss(command);
    std::string token;
    iss >> token;  // setoption
    iss >> token;  // name
    std::string name;
    while (iss >> token && token != "value") {
        if (!name.empty()) {
            name += ' ';
        }
        name += token;
    }

    std::string value;
    if (token == "value") {
        std::getline(iss, value);
        if (!value.empty() && value.front() == ' ') {
            value.erase(value.begin());
        }
    }

    try {
        if (name == "Hash") {
            int mb = std::max(1, std::stoi(value));
            search_.set_table_size_mb(static_cast<std::size_t>(mb));
        } else if (name == "Threads") {
            int threads = std::max(1, std::stoi(value));
            search_.set_threads(threads);
        } else if (name == "Move Overhead") {
            move_overhead_ms_ = std::max(0, std::stoi(value));
        } else if (name == "Base Time Percent") {
            double percent = std::clamp(std::stod(value), 0.0, 100.0);
            time_config_.base_allocation = percent / 100.0;
            search_.set_time_manager(time_config_);
        } else if (name == "Increment Percent") {
            double percent = std::clamp(std::stod(value), 0.0, 500.0);
            time_config_.increment_bonus = percent / 100.0;
            search_.set_time_manager(time_config_);
        } else if (name == "Minimum Think Time") {
            time_config_.min_time_ms = std::max(1, std::stoi(value));
            search_.set_time_manager(time_config_);
        } else if (name == "Maximum Think Time") {
            time_config_.max_time_ms = std::max(time_config_.min_time_ms, std::stoi(value));
            search_.set_time_manager(time_config_);
        } else if (name == "EvalNetwork" || name == "NNUENetworkFile") {
            if (!value.empty()) {
                set_global_network_path(value);
                search_.set_evaluator(global_evaluator());
                std::lock_guard<std::mutex> lock(io_mutex_);
                std::cout << "info string nnue network set to " << value << std::endl;
            }
        } else if (name == "Ponder") {
            // Ponder option acknowledged but handled implicitly.
        }
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        std::cout << "info string Failed to set option " << name << ": " << ex.what() << std::endl;
    }
}

Move UCI::parse_move(const std::string& token) {
    std::vector<Move> moves = MoveGenerator::generate_legal_moves(board_);
    for (const Move& move : moves) {
        if (move_to_string(move) == token) {
            return move;
        }
    }
    throw std::runtime_error("Illegal move received: " + token);
}

void UCI::start_search(SearchLimits limits) {
    stop_search(true);

    for (int color = 0; color < kNumColors; ++color) {
        if (limits.time_left_ms[color] > 0) {
            limits.time_left_ms[color] = std::max(0, limits.time_left_ms[color] - move_overhead_ms_);
        }
    }
    if (limits.move_time_ms > 0) {
        limits.move_time_ms = std::max(0, limits.move_time_ms - move_overhead_ms_);
    }

    current_limits_ = limits;
    stop_flag_.store(false);
    have_result_ = false;
    search_.set_time_manager(time_config_);

    Board board_copy = board_;
    searching_.store(true);
    search_thread_ = std::thread([this, board_copy, limits]() mutable {
        auto callback = [this](const SearchResult& result) { send_info(result); };
        SearchResult result = search_.search(board_copy, limits, stop_flag_, callback);
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_result_ = result;
            have_result_ = true;
        }
        report_bestmove(result);
        searching_.store(false);
    });
}

void UCI::stop_search(bool wait_for_join) {
    stop_flag_.store(true);
    if (wait_for_join) {
        join_thread();
        stop_flag_.store(false);
    }
}

void UCI::join_thread() {
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
    searching_.store(false);
}

void UCI::send_info(const SearchResult& result) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    std::cout << "info depth " << result.depth;
    if (result.seldepth > 0) {
        std::cout << " seldepth " << result.seldepth;
    }

    if (std::abs(result.score) >= kMateThreshold) {
        int mate_moves = (kMateValue - std::abs(result.score) + 1) / 2;
        if (result.score < 0) {
            mate_moves = -mate_moves;
        }
        std::cout << " score mate " << mate_moves;
    } else {
        std::cout << " score cp " << result.score;
    }

    auto elapsed_ms = static_cast<int>(result.elapsed.count());
    if (elapsed_ms < 0) {
        elapsed_ms = 0;
    }
    std::cout << " time " << elapsed_ms;
    std::cout << " nodes " << static_cast<unsigned long long>(result.nodes);
    if (elapsed_ms > 0) {
        std::uint64_t nps = result.nodes * 1000ULL / static_cast<std::uint64_t>(elapsed_ms);
        std::cout << " nps " << static_cast<unsigned long long>(nps);
    }

    if (!result.pv.empty()) {
        std::cout << " pv";
        for (const Move& move : result.pv) {
            std::cout << ' ' << move_to_string(move);
        }
    }

    std::cout << std::endl;
}

void UCI::report_bestmove(const SearchResult& result) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    Move best = result.best_move;
    if (best.from == best.to && best.from == 0) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }

    std::cout << "bestmove " << move_to_string(best);
    if (current_limits_.ponder && result.pv.size() >= 2) {
        std::cout << " ponder " << move_to_string(result.pv[1]);
    }
    std::cout << std::endl;
}

}  // namespace chiron

