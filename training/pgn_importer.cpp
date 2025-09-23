#include "training/pgn_importer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "notation.h"

namespace chiron {

namespace {

std::string strip_comments(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    bool in_brace = false;
    int paren_depth = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '{') {
            in_brace = true;
            continue;
        }
        if (c == '}') {
            in_brace = false;
            continue;
        }
        if (c == '(') {
            ++paren_depth;
            continue;
        }
        if (c == ')') {
            if (paren_depth > 0) {
                --paren_depth;
            }
            continue;
        }
        if (!in_brace && paren_depth == 0) {
            output.push_back(c);
        }
    }
    return output;
}

std::string trim(const std::string& value) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    auto begin = std::find_if_not(value.begin(), value.end(), is_space);
    auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

bool is_move_number(const std::string& token) {
    return !token.empty() && std::isdigit(static_cast<unsigned char>(token.front()));
}

bool is_result_token(const std::string& token) {
    return token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*";
}

}  // namespace

int PgnImporter::result_to_target(const std::string& result_tag) {
    if (result_tag == "1-0") {
        return 1000;
    }
    if (result_tag == "0-1") {
        return -1000;
    }
    if (result_tag == "1/2-1/2") {
        return 0;
    }
    return 0;
}

std::vector<TrainingExample> PgnImporter::import_file(const std::string& path, bool include_draws) const {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open PGN file: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    std::string content = strip_comments(buffer.str());

    std::istringstream iss(content);
    std::string token;
    Board board;
    board.set_start_position();
    std::vector<std::string> positions;
    std::string current_result;
    std::vector<TrainingExample> examples;

    while (iss >> token) {
        if (token.empty()) {
            continue;
        }
        if (token.front() == '[') {
            if (!positions.empty() && !current_result.empty()) {
                int target = result_to_target(current_result);
                if (include_draws || target != 0) {
                    for (const std::string& fen : positions) {
                        examples.push_back({fen, target});
                    }
                }
                positions.clear();
            }
            board.set_start_position();
            current_result.clear();

            std::string tag_name;
            std::string tag_value;
            std::size_t space_pos = token.find(' ');
            if (space_pos != std::string::npos) {
                tag_name = token.substr(1, space_pos - 1);
                tag_value = token.substr(space_pos + 1);
                while (!tag_value.empty() && tag_value.back() != ']') {
                    std::string continuation;
                    if (!(iss >> continuation)) {
                        break;
                    }
                    tag_value += ' ' + continuation;
                }
                if (!tag_value.empty() && tag_value.back() == ']') {
                    tag_value.pop_back();
                }
                tag_value = trim(tag_value);
                if (!tag_value.empty() && tag_value.front() == '"') {
                    tag_value.erase(tag_value.begin());
                }
                if (!tag_value.empty() && tag_value.back() == '"') {
                    tag_value.pop_back();
                }
                if (tag_name == "Result") {
                    current_result = tag_value;
                }
            }
            continue;
        }

        if (is_move_number(token)) {
            continue;
        }

        if (is_result_token(token)) {
            if (!positions.empty()) {
                int target = result_to_target(!current_result.empty() ? current_result : token);
                if (include_draws || target != 0) {
                    for (const std::string& fen : positions) {
                        examples.push_back({fen, target});
                    }
                }
                positions.clear();
            }
            board.set_start_position();
            current_result.clear();
            continue;
        }

        std::string san = trim(token);
        if (san.empty()) {
            continue;
        }

        try {
            TrainingExample example;
            example.fen = board.fen();
            positions.push_back(example.fen);

            Move move = san_to_move(board, san);
            Board::State state;
            board.make_move(move, state);
        } catch (const std::exception&) {
            // Skip malformed moves.
        }
    }

    if (!positions.empty()) {
        int target = result_to_target(current_result);
        if (include_draws || target != 0) {
            for (const std::string& fen : positions) {
                examples.push_back({fen, target});
            }
        }
    }

    return examples;
}

void PgnImporter::write_dataset(const std::string& pgn_path, const std::string& output_path, bool include_draws) const {
    std::vector<TrainingExample> data = import_file(pgn_path, include_draws);
    save_training_file(output_path, data);
}

}  // namespace chiron

