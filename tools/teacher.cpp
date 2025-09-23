#include "tools/teacher.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace chiron {

namespace {

constexpr int kMateValue = 32000;

std::string build_script(const TeacherConfig& config, const std::vector<std::string>& fens) {
    std::ostringstream script;
    script << "uci\n";
    if (config.threads > 1) {
        script << "setoption name Threads value " << config.threads << "\n";
    }
    script << "isready\n";
    for (const std::string& fen : fens) {
        script << "position fen " << fen << "\n";
        script << "go depth " << config.depth << "\n";
    }
    script << "quit\n";
    return script.str();
}

std::string quote_path(const std::filesystem::path& path) {
    std::string str = path.string();
    if (str.find(' ') != std::string::npos) {
        return '"' + str + '"';
    }
    return str;
}

int parse_score_from_line(const std::string& line, int current_score, bool& have_score) {
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        if (token == "score") {
            std::string type;
            if (!(iss >> type)) {
                break;
            }
            if (type == "cp") {
                int cp = 0;
                if (iss >> cp) {
                    current_score = cp;
                    have_score = true;
                }
            } else if (type == "mate") {
                int mate = 0;
                if (iss >> mate) {
                    int sign = mate >= 0 ? 1 : -1;
                    int magnitude = std::abs(mate);
                    current_score = sign * (kMateValue - magnitude * 100);
                    have_score = true;
                }
            }
        }
    }
    return current_score;
}

std::vector<int> parse_output(const std::filesystem::path& output_path, std::size_t expected) {
    std::ifstream stream(output_path);
    if (!stream) {
        throw std::runtime_error("Failed to read teacher engine output from " + output_path.string());
    }
    std::vector<int> results;
    results.reserve(expected);
    std::string line;
    int current_score = 0;
    bool have_score = false;
    while (std::getline(stream, line)) {
        if (line.rfind("info", 0) == 0) {
            current_score = parse_score_from_line(line, current_score, have_score);
        }
        if (line.rfind("bestmove", 0) == 0) {
            results.push_back(have_score ? current_score : 0);
            current_score = 0;
            have_score = false;
            if (results.size() == expected) {
                break;
            }
        }
    }
    return results;
}

std::filesystem::path write_temp_file(const std::string& prefix, const std::string& content) {
    std::filesystem::path directory = std::filesystem::temp_directory_path();
    char buffer[L_tmpnam];
    if (std::tmpnam(buffer) == nullptr) {
        throw std::runtime_error("Failed to generate temporary file name");
    }
    std::string name = prefix + buffer;
    std::replace(name.begin(), name.end(), '\\', '_');
    std::replace(name.begin(), name.end(), '/', '_');
    std::filesystem::path temp = directory / name;
    std::ofstream stream(temp);
    if (!stream) {
        throw std::runtime_error("Failed to open temporary file for writing");
    }
    stream << content;
    return temp;
}

}  // namespace

TeacherEngine::TeacherEngine(TeacherConfig config) : config_(std::move(config)) {}

std::vector<int> TeacherEngine::evaluate(const std::vector<std::string>& fens) const {
    if (config_.engine_path.empty()) {
        throw std::runtime_error("Teacher engine path not configured");
    }
    if (fens.empty()) {
        return {};
    }

    std::string script_content = build_script(config_, fens);
    auto script_path = write_temp_file("chiron-teacher", script_content);
    auto output_path = write_temp_file("chiron-teacher-out", "");

    std::string command;
#ifdef _WIN32
    command = "cmd /C \"" + quote_path(config_.engine_path) + " < " + quote_path(script_path) + " > " + quote_path(output_path) + "\"";
#else
    command = quote_path(config_.engine_path) + " < " + quote_path(script_path) + " > " + quote_path(output_path);
#endif

    int result = std::system(command.c_str());
    if (result != 0) {
        std::filesystem::remove(script_path);
        std::filesystem::remove(output_path);
        throw std::runtime_error("Teacher engine process failed with exit code " + std::to_string(result));
    }

    std::vector<int> scores = parse_output(output_path, fens.size());

    std::filesystem::remove(script_path);
    std::filesystem::remove(output_path);

    if (scores.size() != fens.size()) {
        throw std::runtime_error("Teacher engine returned insufficient evaluations");
    }
    return scores;
}

int TeacherEngine::evaluate_single(const std::string& fen) const {
    std::vector<std::string> fens{fen};
    std::vector<int> scores = evaluate(fens);
    return scores.empty() ? 0 : scores.front();
}

}  // namespace chiron

