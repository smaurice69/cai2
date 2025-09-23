#include "uci.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "evaluation.h"

namespace chiron {

UCI::UCI() : board_(), search_(1 << 20) {}

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") {
            std::cout << "id name Chiron" << std::endl;
            std::cout << "id author OpenAI Assistant" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (line == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (line.rfind("position", 0) == 0) {
            handle_position(line);
        } else if (line.rfind("go", 0) == 0) {
            handle_go(line);
        } else if (line.rfind("setoption", 0) == 0) {
            handle_setoption(line);
        } else if (line == "ucinewgame") {
            board_.set_start_position();
            search_.clear();
        } else if (line == "stop") {
            // Searches are synchronous, so stop simply acknowledges the command.
            std::cout << "info string stop acknowledged" << std::endl;
        } else if (line == "quit") {
            break;
        }
    }
}

void UCI::handle_position(const std::string& command) {
    std::istringstream iss(command);
    std::string token;
    iss >> token;  // consume "position"

    std::vector<std::string> tokens;
    while (iss >> token) {
        tokens.push_back(token);
    }

    if (tokens.empty()) {
        return;
    }

    std::size_t index = 0;
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

void UCI::handle_setoption(const std::string& command) {
    std::istringstream iss(command);
    std::string token;
    iss >> token;  // setoption
    std::string name;
    while (iss >> token && token != "value") {
        if (token == "name") {
            continue;
        }
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

    if (name == "NNUENetworkFile" || name == "EvalNetwork") {
        if (!value.empty()) {
            set_global_network_path(value);
            search_.set_evaluator(global_evaluator());
            std::cout << "info string nnue network set to " << value << std::endl;
        }
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

void UCI::handle_go(const std::string& command) {
    std::istringstream iss(command);
    std::string token;
    iss >> token;  // go

    int depth = 4;
    while (iss >> token) {
        if (token == "depth") {
            if (!(iss >> depth)) {
                depth = 4;
            }
        }
    }

    Move best = search_.search_best_move(board_, depth);
    if (best.from == best.to && best.from == 0 && best.to == 0 && !best.is_promotion()) {
        // No legal move available; signal as required by UCI.
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    std::cout << "bestmove " << move_to_string(best) << std::endl;
}

}  // namespace chiron

