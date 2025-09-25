#include "uci.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluation.h"
#include "perft.h"
#include "tools/teacher.h"
#include "tools/tuning.h"
#include "training/pgn_importer.h"
#include "training/selfplay.h"
#include "training/trainer.h"

namespace {

using chiron::Board;
using chiron::ParameterSet;
using chiron::PgnImporter;
using chiron::TeacherEngine;
using chiron::Trainer;
using chiron::TrainingExample;
using chiron::load_training_file;
using chiron::perft;
using chiron::save_training_file;
using chiron::nnue::kDefaultHiddenSize;

int parse_int(const std::vector<std::string>& args, std::size_t& index, const std::string& option) {
    if (index + 1 >= args.size()) {
        throw std::invalid_argument(option + " requires a value");
    }
    try {
        return std::stoi(args[++index]);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer for " + option);
    }
}

double parse_double(const std::vector<std::string>& args, std::size_t& index, const std::string& option) {
    if (index + 1 >= args.size()) {
        throw std::invalid_argument(option + " requires a value");
    }
    try {
        return std::stod(args[++index]);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid floating point value for " + option);
    }
}

std::size_t parse_size(const std::vector<std::string>& args, std::size_t& index, const std::string& option) {
    if (index + 1 >= args.size()) {
        throw std::invalid_argument(option + " requires a value");
    }
    try {
        return static_cast<std::size_t>(std::stoll(args[++index]));
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid size for " + option);
    }
}

int run_perft(const std::vector<std::string>& args) {
    Board board;
    board.set_start_position();
    int depth = 1;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--depth") {
            depth = parse_int(args, i, opt);
        } else if (opt == "--fen") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--fen requires a value");
            board.set_from_fen(args[++i]);
        }
    }
    if (depth <= 0) {
        throw std::invalid_argument("perft depth must be positive");
    }
    std::uint64_t nodes = perft(board, depth);
    std::cout << "Perft(" << depth << ") = " << nodes << std::endl;
    return 0;
}

int run_selfplay(const std::vector<std::string>& args) {
    chiron::SelfPlayConfig config;
    config.white.name = "Chiron";
    config.black.name = "Chiron";

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--games") {
            config.games = std::max(1, parse_int(args, i, opt));
        } else if (opt == "--depth") {
            int depth = parse_int(args, i, opt);
            config.white.max_depth = depth;
            config.black.max_depth = depth;
        } else if (opt == "--white-depth") {
            config.white.max_depth = parse_int(args, i, opt);
        } else if (opt == "--black-depth") {
            config.black.max_depth = parse_int(args, i, opt);
        } else if (opt == "--white-name") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.white.name = args[++i];
        } else if (opt == "--black-name") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.black.name = args[++i];
        } else if (opt == "--results") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.results_log = args[++i];
        } else if (opt == "--pgn") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.pgn_path = args[++i];
        } else if (opt == "--no-results") {
            config.capture_results = false;
        } else if (opt == "--no-pgn") {
            config.capture_pgn = false;
        } else if (opt == "--record-fens") {
            config.record_fens = true;
        } else if (opt == "--verbose") {
            config.verbose = true;
        } else if (opt == "--verboselite") {
            config.verbose_lite = true;
        } else if (opt == "--max-ply") {
            config.max_ply = parse_int(args, i, opt);
        } else if (opt == "--seed") {
            config.seed = static_cast<unsigned int>(parse_int(args, i, opt));
        } else if (opt == "--table-size") {
            std::size_t size = parse_size(args, i, opt);
            config.white.table_size = size;
            config.black.table_size = size;
        } else if (opt == "--white-table") {
            config.white.table_size = parse_size(args, i, opt);
        } else if (opt == "--black-table") {
            config.black.table_size = parse_size(args, i, opt);
        } else if (opt == "--network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            std::string value = args[++i];
            config.white.network_path = value;
            config.black.network_path = value;
        } else if (opt == "--white-network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.white.network_path = args[++i];
        } else if (opt == "--black-network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.black.network_path = args[++i];
        } else if (opt == "--threads") {
            int threads = parse_int(args, i, opt);
            config.white.threads = threads;
            config.black.threads = threads;
        } else if (opt == "--white-threads") {
            config.white.threads = parse_int(args, i, opt);
        } else if (opt == "--black-threads") {
            config.black.threads = parse_int(args, i, opt);
        } else if (opt == "--fixed-colors") {
            config.alternate_colors = false;
        } else if (opt == "--alternate-colors") {
            config.alternate_colors = true;
        } else if (opt == "--concurrency") {
            config.concurrency = std::max(1, parse_int(args, i, opt));
        } else if (opt == "--enable-training") {
            config.enable_training = true;
        } else if (opt == "--disable-training") {
            config.enable_training = false;
        } else if (opt == "--training-batch") {
            config.training_batch_size = parse_size(args, i, opt);
        } else if (opt == "--training-rate") {
            config.training_learning_rate = parse_double(args, i, opt);
        } else if (opt == "--training-output") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.training_output_path = args[++i];
        } else if (opt == "--training-history") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            config.training_history_dir = args[++i];
        } else if (opt == "--training-hidden") {
            config.training_hidden_size = parse_size(args, i, opt);
        } else {
            throw std::invalid_argument("Unknown selfplay option: " + opt);
        }
    }

    chiron::SelfPlayOrchestrator orchestrator(config);
    orchestrator.run();
    return 0;
}

int run_sprt(const std::vector<std::string>& args) {
    chiron::SelfPlayConfig match_config;
    match_config.games = 1;
    match_config.capture_results = false;
    match_config.capture_pgn = false;
    match_config.white.name = "Baseline";
    match_config.black.name = "Candidate";

    chiron::EngineConfig baseline;
    baseline.name = "Baseline";
    chiron::EngineConfig candidate;
    candidate.name = "Candidate";
    chiron::SprtConfig sprt;

    for (std::size_t i = 2; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--games") {
            sprt.max_games = std::max(1, parse_int(args, i, opt));
        } else if (opt == "--alpha") {
            sprt.alpha = parse_double(args, i, opt);
        } else if (opt == "--beta") {
            sprt.beta = parse_double(args, i, opt);
        } else if (opt == "--elo0") {
            sprt.elo0 = parse_double(args, i, opt);
        } else if (opt == "--elo1") {
            sprt.elo1 = parse_double(args, i, opt);
        } else if (opt == "--draw") {
            sprt.draw_ratio = parse_double(args, i, opt);
        } else if (opt == "--results") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            sprt.results_path = args[++i];
        } else if (opt == "--depth") {
            int depth = parse_int(args, i, opt);
            baseline.max_depth = depth;
            candidate.max_depth = depth;
        } else if (opt == "--baseline-depth") {
            baseline.max_depth = parse_int(args, i, opt);
        } else if (opt == "--candidate-depth") {
            candidate.max_depth = parse_int(args, i, opt);
        } else if (opt == "--network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            std::string value = args[++i];
            baseline.network_path = value;
            candidate.network_path = value;
        } else if (opt == "--baseline-network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            baseline.network_path = args[++i];
        } else if (opt == "--candidate-network") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            candidate.network_path = args[++i];
        } else if (opt == "--baseline-name") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            baseline.name = args[++i];
            match_config.white.name = baseline.name;
        } else if (opt == "--candidate-name") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            candidate.name = args[++i];
            match_config.black.name = candidate.name;
        } else if (opt == "--table-size") {
            std::size_t size = parse_size(args, i, opt);
            baseline.table_size = size;
            candidate.table_size = size;
        } else {
            throw std::invalid_argument("Unknown sprt option: " + opt);
        }
    }

    chiron::SprtTester tester(match_config, baseline, candidate, sprt);
    chiron::SprtSummary summary = tester.run();

    std::cout << "SPRT conclusion: " << summary.conclusion << "\n";
    std::cout << "Games: " << summary.games_played << ", candidate wins: " << summary.candidate_wins
              << ", baseline wins: " << summary.baseline_wins << ", draws: " << summary.draws << "\n";
    std::cout << "LLR: " << summary.llr << "\n";
    if (summary.elo) {
        std::cout << "Estimated Elo: " << std::fixed << std::setprecision(2) << *summary.elo;
        if (summary.elo_confidence) {
            std::cout << " ±" << *summary.elo_confidence;
        }
        std::cout << std::defaultfloat << std::setprecision(6) << "\n";
    }
    return 0;
}

int run_time_analysis(const std::vector<std::string>& args) {
    chiron::TimeHeuristicConfig config;
    std::string log_path;

    for (std::size_t i = 2; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--log") {
            if (i + 1 >= args.size()) throw std::invalid_argument(opt + " requires a value");
            log_path = args[++i];
        } else if (opt == "--base") {
            config.base_allocation = parse_double(args, i, opt);
        } else if (opt == "--increment") {
            config.increment_bonus = parse_double(args, i, opt);
        } else if (opt == "--min") {
            config.min_time_ms = parse_int(args, i, opt);
        } else if (opt == "--max") {
            config.max_time_ms = parse_int(args, i, opt);
        } else {
            throw std::invalid_argument("Unknown time tuning option: " + opt);
        }
    }

    if (log_path.empty()) {
        throw std::invalid_argument("--log is required for time analysis");
    }

    chiron::TimeManager manager(config);
    chiron::TimeTuningReport report = manager.analyse_results_log(log_path);

    std::cout << "Analysed games: " << report.games_evaluated << "\n";
    std::cout << "Average ply: " << report.average_ply << "\n";
    std::cout << "Recommended moves-to-go: " << report.recommended_moves_to_go << "\n";

    int sample = manager.allocate_time_ms(60000, 0, 20, static_cast<int>(report.recommended_moves_to_go));
    std::cout << "Sample allocation with 60s remaining: " << sample << " ms" << std::endl;
    return 0;
}

int run_train_command(const std::vector<std::string>& args) {
    std::string input_path;
    std::string output_path = "trained.nnue";
    double learning_rate = 0.05;
    std::size_t batch_size = 256;
    std::size_t hidden_size = kDefaultHiddenSize;
    int iterations = 1;
    bool shuffle = false;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--input") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--input requires a path");
            input_path = args[++i];
        } else if (opt == "--output") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--output requires a path");
            output_path = args[++i];
        } else if (opt == "--rate") {
            learning_rate = parse_double(args, i, opt);
        } else if (opt == "--batch") {
            batch_size = parse_size(args, i, opt);
        } else if (opt == "--iterations") {
            iterations = parse_int(args, i, opt);
        } else if (opt == "--shuffle") {
            shuffle = true;
        } else if (opt == "--hidden") {
            hidden_size = parse_size(args, i, opt);
        }
    }

    if (input_path.empty()) {
        throw std::invalid_argument("train command requires --input dataset path");
    }

    std::vector<TrainingExample> data = load_training_file(input_path);
    if (data.empty()) {
        throw std::runtime_error("No training samples loaded from " + input_path);
    }

    if (shuffle) {
        std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
        std::shuffle(data.begin(), data.end(), rng);
    }

    ParameterSet parameters(hidden_size);
    if (!output_path.empty() && std::filesystem::exists(output_path)) {
        parameters.load(output_path);
    }

    Trainer trainer({learning_rate, 0.0005});
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t offset = 0; offset < data.size(); offset += batch_size) {
            std::size_t end = std::min(offset + batch_size, data.size());
            std::vector<TrainingExample> batch(data.begin() + static_cast<std::ptrdiff_t>(offset),
                                              data.begin() + static_cast<std::ptrdiff_t>(end));
            trainer.train_batch(batch, parameters);
        }
    }

    if (!output_path.empty()) {
        parameters.save(output_path);
    }
    return 0;
}

int run_import_pgn(const std::vector<std::string>& args) {
    std::string pgn_path;
    std::string output_path = "dataset.txt";
    bool include_draws = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--pgn") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--pgn requires a file path");
            pgn_path = args[++i];
        } else if (opt == "--output") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--output requires a file path");
            output_path = args[++i];
        } else if (opt == "--no-draws") {
            include_draws = false;
        }
    }

    if (pgn_path.empty()) {
        throw std::invalid_argument("import-pgn requires --pgn input file");
    }

    PgnImporter importer;
    std::vector<TrainingExample> examples = importer.import_file(pgn_path, include_draws);
    save_training_file(output_path, examples);
    std::cout << "Wrote " << examples.size() << " training samples to " << output_path << std::endl;
    return 0;
}

int run_teacher_command(const std::vector<std::string>& args) {
    std::string engine_path;
    std::string positions_path;
    std::string output_path = "teacher_labels.txt";
    int depth = 20;
    int threads = 1;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& opt = args[i];
        if (opt == "--engine") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--engine requires a path");
            engine_path = args[++i];
        } else if (opt == "--positions") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--positions requires a file path");
            positions_path = args[++i];
        } else if (opt == "--output") {
            if (i + 1 >= args.size()) throw std::invalid_argument("--output requires a file path");
            output_path = args[++i];
        } else if (opt == "--depth") {
            depth = parse_int(args, i, opt);
        } else if (opt == "--threads") {
            threads = parse_int(args, i, opt);
        }
    }

    if (engine_path.empty()) {
        throw std::invalid_argument("teacher command requires --engine path");
    }
    if (positions_path.empty()) {
        throw std::invalid_argument("teacher command requires --positions file");
    }

    std::ifstream positions_stream(positions_path);
    if (!positions_stream) {
        throw std::runtime_error("Failed to open positions file: " + positions_path);
    }
    std::vector<std::string> fens;
    std::string line;
    while (std::getline(positions_stream, line)) {
        if (!line.empty()) {
            fens.push_back(line);
        }
    }
    if (fens.empty()) {
        throw std::runtime_error("Positions file is empty");
    }

    TeacherEngine teacher({engine_path, depth, threads});
    std::vector<int> scores = teacher.evaluate(fens);
    std::vector<TrainingExample> examples;
    examples.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        examples.push_back({fens[i], scores[i]});
    }
    save_training_file(output_path, examples);
    std::cout << "Annotated " << examples.size() << " positions to " << output_path << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (args.empty()) {
            chiron::UCI uci;
            uci.loop();
            return 0;
        }

        const std::string& command = args[0];
        if (command == "selfplay") {
            return run_selfplay(args);
        }
        if (command == "perft") {
            return run_perft(args);
        }
        if (command == "train") {
            return run_train_command(args);
        }
        if (command == "import-pgn") {
            return run_import_pgn(args);
        }
        if (command == "teacher") {
            return run_teacher_command(args);
        }
        if (command == "tune") {
            if (args.size() < 2) {
                throw std::invalid_argument("tune requires a subcommand (sprt/time)");
            }
            const std::string& sub = args[1];
            if (sub == "sprt") {
                return run_sprt(args);
            }
            if (sub == "time") {
                return run_time_analysis(args);
            }
            throw std::invalid_argument("Unknown tune subcommand: " + sub);
        }

        throw std::invalid_argument("Unknown command: " + command);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
