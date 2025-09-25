#include "training/selfplay.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <ctime>
#include <iomanip>
#include <ios>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "bitboard.h"
#include "evaluation.h"
#include "movegen.h"

namespace chiron {

namespace {

bool is_null_move(const Move& move) {
    return move.from == 0 && move.to == 0 && move.promotion == PieceType::None && move.flags == MoveFlag::Quiet;
}

char piece_to_char(PieceType piece) {
    switch (piece) {
        case PieceType::Knight:
            return 'N';
        case PieceType::Bishop:
            return 'B';
        case PieceType::Rook:
            return 'R';
        case PieceType::Queen:
            return 'Q';
        case PieceType::King:
            return 'K';
        case PieceType::Pawn:
        case PieceType::None:
        default:
            return '\0';
    }
}

bool is_light_square(int square) {
    int file = square & 7;
    int rank = square >> 3;
    return (file + rank) % 2 == 0;
}

bool insufficient_material(const Board& board) {
    Bitboard white_minors = board.pieces(Color::White, PieceType::Bishop) | board.pieces(Color::White, PieceType::Knight);
    Bitboard black_minors = board.pieces(Color::Black, PieceType::Bishop) | board.pieces(Color::Black, PieceType::Knight);

    Bitboard white_majors = board.pieces(Color::White, PieceType::Queen) | board.pieces(Color::White, PieceType::Rook) |
                            board.pieces(Color::White, PieceType::Pawn);
    Bitboard black_majors = board.pieces(Color::Black, PieceType::Queen) | board.pieces(Color::Black, PieceType::Rook) |
                            board.pieces(Color::Black, PieceType::Pawn);

    if (white_majors || black_majors) {
        return false;
    }

    int white_minors_count = popcount(white_minors);
    int black_minors_count = popcount(black_minors);

    if (white_minors_count == 0 && black_minors_count == 0) {
        return true;  // King vs King
    }
    if ((white_minors_count <= 1) && (black_minors_count == 0)) {
        return true;  // King and minor vs King
    }
    if ((black_minors_count <= 1) && (white_minors_count == 0)) {
        return true;
    }

    if (white_minors_count == 1 && black_minors_count == 1 &&
        board.pieces(Color::White, PieceType::Bishop) && board.pieces(Color::Black, PieceType::Bishop)) {
        Bitboard wb = board.pieces(Color::White, PieceType::Bishop);
        Bitboard bb = board.pieces(Color::Black, PieceType::Bishop);
        int white_square = wb ? pop_lsb(wb) : -1;
        int black_square = bb ? pop_lsb(bb) : -1;
        if (white_square != -1 && black_square != -1 && is_light_square(white_square) == is_light_square(black_square)) {
            return true;
        }
    }

    return false;
}

std::string escape_json(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\':
                escaped += "\\\\";
                break;
            case '\"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return escaped;
}

std::string join_string_array(const std::vector<std::string>& values) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << '\"' << escape_json(values[i]) << '\"';
    }
    oss << ']';
    return oss.str();
}

std::string format_moves(const std::vector<std::string>& moves) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (i % 2 == 0) {
            oss << static_cast<int>(i / 2 + 1) << ". ";
        }
        oss << moves[i];
        if (i + 1 < moves.size()) {
            oss << ' ';
        }
    }
    return oss.str();
}

std::string move_to_san(Board& board, const Move& move) {
    if (move.is_castle()) {
        return (move.flags & MoveFlag::KingCastle) ? "O-O" : "O-O-O";
    }

    PieceType moving_piece = board.piece_type_at(move.from);
    std::string san;
    if (moving_piece != PieceType::Pawn) {
        san += piece_to_char(moving_piece);

        auto legal_moves = MoveGenerator::generate_legal_moves(board);
        bool needs_file = false;
        bool needs_rank = false;
        bool conflict = false;
        for (const Move& candidate : legal_moves) {
            if (candidate.to == move.to && candidate.from != move.from) {
                PieceType candidate_piece = board.piece_type_at(candidate.from);
                if (candidate_piece == moving_piece) {
                    conflict = true;
                    if ((candidate.from & 7) == (move.from & 7)) {
                        needs_file = true;
                    }
                    if ((candidate.from >> 3) == (move.from >> 3)) {
                        needs_rank = true;
                    }
                }
            }
        }
        if (conflict) {
            if (!needs_file) {
                san += static_cast<char>('a' + (move.from & 7));
            } else if (!needs_rank) {
                san += static_cast<char>('1' + (move.from >> 3));
            } else {
                san += static_cast<char>('a' + (move.from & 7));
                san += static_cast<char>('1' + (move.from >> 3));
            }
        }
    } else if (move.is_capture()) {
        san += static_cast<char>('a' + (move.from & 7));
    }

    if (move.is_capture()) {
        san += 'x';
    }
    san += square_to_string(static_cast<Square>(move.to));

    if (move.is_promotion()) {
        san += '=';
        san += piece_to_char(move.promotion);
    }

    Board::State state;
    board.make_move(move, state);
    bool opponent_in_check = board.in_check(board.side_to_move());
    bool opponent_has_moves = !MoveGenerator::generate_legal_moves(board).empty();
    board.undo_move(move, state);

    if (opponent_in_check) {
        san += opponent_has_moves ? '+' : '#';
    }

    return san;
}

std::shared_ptr<nnue::Evaluator> create_evaluator(const EngineConfig& config) {
    auto evaluator = std::make_shared<nnue::Evaluator>();
    if (!config.network_path.empty()) {
        evaluator->set_network_path(config.network_path);
    }
    return evaluator;
}

constexpr int kMateValue = 32000;
constexpr int kMateThreshold = kMateValue - 512;

std::string color_name(Color color) {
    return color == Color::White ? "White" : "Black";
}

std::string format_evaluation(int score, Color mover) {
    std::ostringstream oss;
    if (std::abs(score) >= kMateThreshold) {
        int mate_moves = (kMateValue - std::abs(score) + 1) / 2;
        Color winner = score > 0 ? mover : opposite_color(mover);
        if (score < 0) {
            oss << "-M" << mate_moves;
        } else {
            oss << "+M" << mate_moves;
        }
        oss << " (" << color_name(winner) << " mates in " << mate_moves << ')';
    } else {
        double pawns = static_cast<double>(score) / 100.0;
        oss << std::showpos << std::fixed << std::setprecision(2) << pawns << std::noshowpos;
        oss << " (" << score << " cp for " << color_name(mover) << ')';
    }
    return oss.str();
}

std::string format_pv(Board board, const std::vector<Move>& pv) {
    if (pv.empty()) {
        return std::string{};
    }
    std::ostringstream oss;
    bool first = true;
    for (const Move& move : pv) {
        if (!first) {
            oss << ' ';
        }
        std::string san = move_to_san(board, move);
        oss << san;
        Board::State state;
        board.make_move(move, state);
        first = false;
    }
    return oss.str();
}

}  // namespace

SelfPlayOrchestrator::SelfPlayOrchestrator(SelfPlayConfig config)
    : config_(std::move(config)),
      rng_(config_.seed != 0U ? config_.seed : static_cast<unsigned int>(std::random_device{}())),
      trainer_(Trainer::Config{config_.training_learning_rate, 0.0005}) {
    if (!config_.training_output_path.empty()) {
        std::filesystem::path output_path(config_.training_output_path);
        training_history_prefix_ = output_path.stem().string();
        training_history_extension_ = output_path.extension().string();
    }
    if (training_history_prefix_.empty()) {
        training_history_prefix_ = "chiron-selfplay";
    }
    if (training_history_extension_.empty()) {
        training_history_extension_ = ".nnue";
    }

    if (config_.enable_training) {
        config_.record_fens = true;
        if (!config_.training_output_path.empty() && std::filesystem::exists(config_.training_output_path)) {
            parameters_.load(config_.training_output_path);
            set_global_network_path(config_.training_output_path);
        }
        training_buffer_.reserve(config_.training_batch_size);
        if (config_.white.network_path.empty() && !config_.training_output_path.empty() &&
            std::filesystem::exists(config_.training_output_path)) {
            config_.white.network_path = config_.training_output_path;
        }
        if (config_.black.network_path.empty() && !config_.training_output_path.empty() &&
            std::filesystem::exists(config_.training_output_path)) {
            config_.black.network_path = config_.training_output_path;
        }
        training_iteration_ = detect_existing_history_iteration();

      
        total_positions_trained_ = static_cast<std::size_t>(training_iteration_) * config_.training_batch_size;
        total_positions_collected_ = total_positions_trained_;
    }
}

void SelfPlayOrchestrator::ensure_streams() {
    if (streams_open_) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (streams_open_) {
        return;
    }
    std::ios_base::openmode mode = std::ios::out;
    mode |= config_.append_logs ? std::ios::app : std::ios::trunc;
    if (config_.capture_results && !config_.results_log.empty()) {
        std::filesystem::path results_path(config_.results_log);
        if (results_path.has_parent_path()) {
            std::filesystem::create_directories(results_path.parent_path());
        }
        results_stream_.open(results_path, mode);
    }
    if (config_.capture_pgn && !config_.pgn_path.empty()) {
        std::filesystem::path pgn_path(config_.pgn_path);
        if (pgn_path.has_parent_path()) {
            std::filesystem::create_directories(pgn_path.parent_path());
        }
        pgn_stream_.open(pgn_path, mode);
    }
    streams_open_ = true;
}

void SelfPlayOrchestrator::run() {
    ensure_streams();
    int total_games = config_.games;
    int concurrency = std::max(1, config_.concurrency);
    if (config_.verbose) {
        std::ostringstream header;
        header << "[SelfPlay] Starting " << total_games << " game(s) with concurrency " << concurrency
               << ". Max ply " << config_.max_ply << '.';
        log_verbose(header.str());

        std::ostringstream engines;
        engines << "[SelfPlay] White " << config_.white.name << " (depth " << config_.white.max_depth
                << ", threads " << config_.white.threads << ", net "
                << (config_.white.network_path.empty() ? "<default>" : config_.white.network_path) << ") | Black "
                << config_.black.name << " (depth " << config_.black.max_depth << ", threads "
                << config_.black.threads << ", net "
                << (config_.black.network_path.empty() ? "<default>" : config_.black.network_path) << ')';
        log_verbose(engines.str());

        if (config_.enable_training) {
            std::ostringstream train;
            train << "[Train] Batch size " << config_.training_batch_size << ", learning rate "
                  << config_.training_learning_rate;
            if (!config_.training_output_path.empty()) {
                train << ", output " << config_.training_output_path;
            } else {
                train << ", output <none>";
            }
            if (!config_.training_history_dir.empty()) {
                train << ", history " << config_.training_history_dir;
            }
            train << ". Previously trained positions " << total_positions_trained_ << '.';
            log_verbose(train.str());
        }
    }
    std::atomic<int> next_game{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(concurrency));
    for (int thread_index = 0; thread_index < concurrency; ++thread_index) {
        workers.emplace_back([this, total_games, &next_game]() {
            while (true) {
                int game = next_game.fetch_add(1);
                if (game >= total_games) {
                    break;
                }
                EngineConfig white;
                EngineConfig black;
                bool alternate_colors = false;
                {
                    std::lock_guard<std::mutex> config_lock(config_mutex_);
                    white = config_.white;
                    black = config_.black;
                    alternate_colors = config_.alternate_colors;
                }
                if (alternate_colors && (game % 2 == 1)) {
                    std::swap(white, black);
                }
                play_game(game, white, black, true);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

SelfPlayResult SelfPlayOrchestrator::play_game(int game_index, const EngineConfig& white, const EngineConfig& black,
                                               bool log_outputs) {
    ensure_streams();
    if (config_.verbose) {
        std::ostringstream start;
        start << "[Game " << (game_index + 1) << "] Start: " << white.name << " (White, depth "
              << white.max_depth << ", threads " << white.threads << ", net "
              << (white.network_path.empty() ? "<default>" : white.network_path) << ") vs " << black.name
              << " (Black, depth " << black.max_depth << ", threads " << black.threads << ", net "
              << (black.network_path.empty() ? "<default>" : black.network_path) << ')';
        log_verbose(start.str());
    }
    SelfPlayResult result = play_single_game(game_index, white, black);
    if (log_outputs) {
        if (config_.capture_results && results_stream_) {
            log_result(game_index, result);
        }
        if (config_.capture_pgn && pgn_stream_) {
            write_pgn(game_index, result);
        }
    }
    handle_training(result);
    if (config_.verbose) {
        std::ostringstream summary;
        summary << "[Game " << (game_index + 1) << "] Final: " << result.result << " (" << result.termination
                << ") after " << result.ply_count << " ply in " << std::fixed << std::setprecision(2)
                << (result.duration_ms / 1000.0) << "s";
        if (config_.enable_training) {
            std::size_t added = 1 + result.fens.size();
            summary << ". Positions collected " << added << " (total collected " << total_positions_collected_
                    << ", trained " << total_positions_trained_ << ")";
        }
        log_verbose(summary.str());
    }
    return result;
}

SelfPlayResult SelfPlayOrchestrator::play_single_game(int game_index, const EngineConfig& white,
                                                      const EngineConfig& black) {
    Board board;
    board.set_start_position();

    SelfPlayResult result;
    result.white_player = white.name;
    result.black_player = black.name;
    result.start_fen = board.fen();

    auto white_eval = create_evaluator(white);
    auto black_eval = create_evaluator(black);

    Search white_search(white.table_size, white_eval);
    Search black_search(black.table_size, black_eval);
    white_search.set_threads(white.threads);
    black_search.set_threads(black.threads);
    white_search.clear();
    black_search.clear();

    auto start_time = std::chrono::steady_clock::now();

    std::unordered_map<std::uint64_t, int> repetition;
    repetition[board.zobrist_key()] = 1;

    int ply = 0;
    bool finished = false;

    while (!finished) {
        if (config_.max_ply > 0 && ply >= config_.max_ply) {
            result.result = "1/2-1/2";
            result.termination = "max-ply";
            break;
        }

        const EngineConfig& cfg = board.side_to_move() == Color::White ? white : black;
        Search& current_search = board.side_to_move() == Color::White ? white_search : black_search;

        SearchLimits limits;
        limits.max_depth = cfg.max_depth;
        if (config_.verbose) {
            int move_number = ply / 2 + 1;
            std::ostringstream search_msg;
            search_msg << "[Game " << (game_index + 1) << "] Searching " << move_number
                       << (board.side_to_move() == Color::White ? ". " : "... ")
                       << (board.side_to_move() == Color::White ? white.name : black.name)
                       << " at depth " << cfg.max_depth;
            if (cfg.threads > 1) {
                search_msg << " (threads " << cfg.threads << ')';
            }
            log_verbose(search_msg.str());
        }
        std::atomic<bool> stop_flag{false};
        InfoCallback info_cb;
        if (config_.verbose) {
            Color mover = board.side_to_move();
            info_cb = [this, game_index, mover, &board](const SearchResult& info) {
                std::ostringstream info_msg;
                info_msg << "[Game " << (game_index + 1) << "] info depth " << info.depth;
                info_msg << " | eval " << format_evaluation(info.score, mover);
                info_msg << " | nodes " << static_cast<unsigned long long>(info.nodes);
                if (info.elapsed.count() > 0) {
                    double elapsed_ms = static_cast<double>(info.elapsed.count());
                    info_msg << " | time " << static_cast<long long>(info.elapsed.count()) << "ms";
                    if (elapsed_ms > 0.0) {
                        double nodes_per_second = static_cast<double>(info.nodes) * 1000.0 / elapsed_ms;
                        if (nodes_per_second > 0.0) {
                            info_msg << " | nps " << static_cast<unsigned long long>(nodes_per_second);
                        }
                    }
                }
                Board pv_board = board;
                std::string pv_line = format_pv(pv_board, info.pv);
                if (!pv_line.empty()) {
                    info_msg << " | pv " << pv_line;
                }
                log_verbose(info_msg.str());
            };
        }
        SearchResult search_result;
        if (info_cb) {
            search_result = current_search.search(board, limits, stop_flag, info_cb);
        } else {
            search_result = current_search.search(board, limits);
        }

        Move best = search_result.best_move;
        if (is_null_move(best)) {
            bool in_check = board.in_check(board.side_to_move());
            if (in_check) {
                result.result = (board.side_to_move() == Color::White) ? "0-1" : "1-0";
                result.termination = "checkmate";
            } else {
                result.result = "1/2-1/2";
                result.termination = "stalemate";
            }
            break;
        }

        std::string san = move_to_san(board, best);
        if (config_.verbose) {
            Board pv_board = board;
            std::string pv_san = format_pv(pv_board, search_result.pv);
            double elapsed_ms = static_cast<double>(search_result.elapsed.count());
            std::uint64_t nps = 0;
            if (elapsed_ms > 0.0) {
                nps = static_cast<std::uint64_t>(static_cast<double>(search_result.nodes) * 1000.0 / elapsed_ms);
            }
            int move_number = ply / 2 + 1;
            Color mover = board.side_to_move();
            const std::string& player_name = (mover == Color::White) ? white.name : black.name;
            std::ostringstream move_log;
            move_log << "[Game " << (game_index + 1) << "] " << move_number
                     << (mover == Color::White ? ". " : "... ") << player_name << " (" << color_name(mover)
                     << ") plays " << san;
            move_log << " | eval " << format_evaluation(search_result.score, mover);
            move_log << " | depth " << search_result.depth;
            if (search_result.seldepth > 0) {
                move_log << " (sel " << search_result.seldepth << ')';
            }
            move_log << " | nodes " << static_cast<unsigned long long>(search_result.nodes);
            if (elapsed_ms > 0.0) {
                move_log << " | time " << static_cast<long long>(search_result.elapsed.count()) << "ms";
            }
            if (nps > 0) {
                move_log << " | nps " << static_cast<unsigned long long>(nps);
            }
            if (!pv_san.empty()) {
                move_log << " | pv " << pv_san;
            }
            log_verbose(move_log.str());
        }
        result.moves_san.push_back(std::move(san));

        Board::State state;
        board.make_move(best, state);
        ++ply;

        repetition[board.zobrist_key()] += 1;

        if (config_.record_fens) {
            result.fens.push_back(board.fen());
        }

        if (board.halfmove_clock() >= 100) {
            result.result = "1/2-1/2";
            result.termination = "fifty-move-rule";
            finished = true;
        } else if (repetition[board.zobrist_key()] >= 3) {
            result.result = "1/2-1/2";
            result.termination = "threefold-repetition";
            finished = true;
        } else if (insufficient_material(board)) {
            result.result = "1/2-1/2";
            result.termination = "insufficient-material";
            finished = true;
        }
    }

    if (result.result.empty()) {
        result.result = "1/2-1/2";
        result.termination = "draw";
    }

    result.end_fen = board.fen();
    result.ply_count = static_cast<int>(result.moves_san.size());
    auto end_time = std::chrono::steady_clock::now();
    result.duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

void SelfPlayOrchestrator::log_result(int game_index, const SelfPlayResult& result) {
    if (!results_stream_) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mutex_);
    results_stream_ << '{';
    results_stream_ << "\"game\":" << (game_index + 1) << ',';
    results_stream_ << "\"white\":\"" << escape_json(result.white_player) << '"' << ',';
    results_stream_ << "\"black\":\"" << escape_json(result.black_player) << '"' << ',';
    results_stream_ << "\"result\":\"" << escape_json(result.result) << '"' << ',';
    results_stream_ << "\"termination\":\"" << escape_json(result.termination) << '"' << ',';
    results_stream_ << "\"ply_count\":" << result.ply_count << ',';
    results_stream_ << "\"duration_ms\":" << std::fixed << std::setprecision(2) << result.duration_ms << ',';
    results_stream_ << "\"start_fen\":\"" << escape_json(result.start_fen) << '"' << ',';
    results_stream_ << "\"end_fen\":\"" << escape_json(result.end_fen) << '"' << ',';

    results_stream_ << "\"moves\":" << join_string_array(result.moves_san);
    if (config_.record_fens) {
        results_stream_ << ",\"fens\":" << join_string_array(result.fens);
    }
    results_stream_ << "}\n" << std::defaultfloat;
    results_stream_.flush();
}

void SelfPlayOrchestrator::write_pgn(int game_index, const SelfPlayResult& result) {
    if (!pgn_stream_) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mutex_);
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time);

    pgn_stream_ << "[Event \"Chiron Self-Play\"]\n";
    pgn_stream_ << "[Site \"Local\"]\n";
    pgn_stream_ << "[Date \"" << std::put_time(&tm, "%Y.%m.%d") << "\"]\n";
    pgn_stream_ << "[Round \"" << (game_index + 1) << "\"]\n";
    pgn_stream_ << "[White \"" << result.white_player << "\"]\n";
    pgn_stream_ << "[Black \"" << result.black_player << "\"]\n";
    pgn_stream_ << "[Result \"" << result.result << "\"]\n";
    pgn_stream_ << "[Termination \"" << result.termination << "\"]\n";
    pgn_stream_ << "[PlyCount \"" << result.ply_count << "\"]\n";
    pgn_stream_ << "[FEN \"" << result.start_fen << "\"]\n";
    pgn_stream_ << "[SetUp \"1\"]\n\n";

    pgn_stream_ << format_moves(result.moves_san);
    if (!result.moves_san.empty()) {
        pgn_stream_ << ' ';
    }
    pgn_stream_ << result.result << "\n\n";
    pgn_stream_.flush();
}

void SelfPlayOrchestrator::handle_training(const SelfPlayResult& result) {
    if (!config_.enable_training) {
        return;
    }

    int target = 0;
    if (result.result == "1-0") {
        target = 1000;
    } else if (result.result == "0-1") {
        target = -1000;
    }

    std::lock_guard<std::mutex> lock(training_mutex_);
    training_buffer_.push_back({result.start_fen, target});
    for (const std::string& fen : result.fens) {
        training_buffer_.push_back({fen, target});
    }

    std::size_t added = 1 + result.fens.size();
    total_positions_collected_ += added;
    if (config_.verbose) {
        std::ostringstream collect;
        collect << "[Train] Collected " << added << " positions (buffer " << training_buffer_.size() << '/'
                << config_.training_batch_size << ", total collected " << total_positions_collected_ << ')';
        log_verbose(collect.str());
    }

    if (training_buffer_.size() >= config_.training_batch_size) {
        std::size_t batch = training_buffer_.size();
        trainer_.train_batch(training_buffer_, parameters_);
        training_buffer_.clear();
        total_positions_trained_ += batch;
        ++training_iteration_;

        std::string updated_network_path;
        std::string snapshot_path;
        if (!config_.training_output_path.empty()) {
            std::filesystem::path output_path(config_.training_output_path);
            if (output_path.has_parent_path()) {
                std::filesystem::create_directories(output_path.parent_path());
            }
            parameters_.save(output_path.string());
            set_global_network_path(output_path.string());
            {
                std::lock_guard<std::mutex> config_lock(config_mutex_);
                config_.white.network_path = output_path.string();
                config_.black.network_path = output_path.string();
            }

          
            updated_network_path = output_path.string();

            if (!config_.training_history_dir.empty()) {
                std::filesystem::path history_dir(config_.training_history_dir);
                std::filesystem::create_directories(history_dir);
                std::ostringstream name;
                name << training_history_prefix_ << "-iter" << std::setw(6) << std::setfill('0')
                     << training_iteration_;
                std::filesystem::path snapshot = history_dir / (name.str() + training_history_extension_);
                parameters_.save(snapshot.string());

              
                snapshot_path = snapshot.string();
            }
        }

        if (config_.verbose) {
            std::ostringstream train_msg;
            train_msg << "[Train] Iteration " << training_iteration_ << " trained on " << batch
                      << " positions (total trained " << total_positions_trained_ << ')';
            if (!updated_network_path.empty()) {
                train_msg << ". Updated network: " << updated_network_path;
            } else {
                train_msg << ". Updated in-memory weights (no output path).";
            }
            log_verbose(train_msg.str());
            if (!snapshot_path.empty()) {
                std::ostringstream snapshot_msg;
                snapshot_msg << "[Train] Snapshot saved to " << snapshot_path;
                log_verbose(snapshot_msg.str());
            }
        }
    }
}

void SelfPlayOrchestrator::log_verbose(const std::string& message) {
    if (!config_.verbose) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::cout << message << std::endl;

  
}

int SelfPlayOrchestrator::detect_existing_history_iteration() const {
    if (config_.training_history_dir.empty()) {
        return 0;
    }
    std::filesystem::path history_dir(config_.training_history_dir);
    if (!std::filesystem::exists(history_dir)) {
        return 0;
    }
    const std::string prefix = training_history_prefix_ + "-iter";
    int max_iter = 0;
    for (const auto& entry : std::filesystem::directory_iterator(history_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!training_history_extension_.empty() &&
            entry.path().extension().string() != training_history_extension_) {
            continue;
        }
        std::string stem = entry.path().stem().string();
        if (stem.rfind(prefix, 0) != 0) {
            continue;
        }
        std::string digits = stem.substr(prefix.size());
        if (digits.empty()) {
            continue;
        }
        try {
            int value = std::stoi(digits);
            if (value > max_iter) {
                max_iter = value;
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    return max_iter;
}

}  // namespace chiron
