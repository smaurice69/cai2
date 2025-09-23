#include "training/selfplay.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <ios>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "bitboard.h"
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

}  // namespace

SelfPlayOrchestrator::SelfPlayOrchestrator(SelfPlayConfig config)
    : config_(std::move(config)),
      rng_(config_.seed != 0U ? config_.seed : static_cast<unsigned int>(std::random_device{}())) {}

void SelfPlayOrchestrator::ensure_streams() {
    if (streams_open_) {
        return;
    }
    std::ios_base::openmode mode = std::ios::out;
    mode |= config_.append_logs ? std::ios::app : std::ios::trunc;
    if (config_.capture_results && !config_.results_log.empty()) {
        results_stream_.open(config_.results_log, mode);
    }
    if (config_.capture_pgn && !config_.pgn_path.empty()) {
        pgn_stream_.open(config_.pgn_path, mode);
    }
    streams_open_ = true;
}

void SelfPlayOrchestrator::run() {
    ensure_streams();
    for (int game = 0; game < config_.games; ++game) {
        EngineConfig white = config_.white;
        EngineConfig black = config_.black;
        if (config_.alternate_colors && (game % 2 == 1)) {
            std::swap(white, black);
        }
        play_game(game, white, black, true);
    }
}

SelfPlayResult SelfPlayOrchestrator::play_game(int game_index, const EngineConfig& white, const EngineConfig& black,
                                               bool log_outputs) {
    ensure_streams();
    SelfPlayResult result = play_single_game(game_index, white, black);
    if (log_outputs) {
        if (config_.capture_results && results_stream_) {
            log_result(game_index, result);
        }
        if (config_.capture_pgn && pgn_stream_) {
            write_pgn(game_index, result);
        }
    }
    return result;
}

SelfPlayResult SelfPlayOrchestrator::play_single_game(int /*game_index*/, const EngineConfig& white,
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

        Move best = current_search.search_best_move(board, cfg.max_depth);
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

}  // namespace chiron
