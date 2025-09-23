#include "board.h"

#include <bit>
#include <cctype>
#include <sstream>
#include <stdexcept>

#include "attacks.h"
#include "zobrist.h"

namespace chiron {

namespace {
constexpr const char* kStartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

int square_from_file_rank(char file, char rank) {
    int f = file - 'a';
    int r = rank - '1';
    return r * 8 + f;
}

}  // namespace

Board::Board() {
    init_attack_tables();
    set_start_position();
}

void Board::clear() {
    for (auto& color_bb : pieces_) {
        color_bb.fill(kEmpty);
    }
    occupancies_.fill(kEmpty);
    occupancy_all_ = kEmpty;
    mailbox_.fill(kEmptySquare);
    side_to_move_ = Color::White;
    castling_rights_ = 0;
    en_passant_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    zobrist_key_ = 0ULL;
}

void Board::place_piece(Color color, PieceType type, int square) {
    Bitboard bb = square_bb(static_cast<Square>(square));
    pieces_[static_cast<int>(color)][static_cast<int>(type)] |= bb;
    occupancies_[static_cast<int>(color)] |= bb;
    occupancy_all_ |= bb;
    mailbox_[square] = encode_piece(color, type);
    zobrist_key_ ^= Zobrist::piece_key(color, type, square);
}

void Board::remove_piece(Color color, PieceType type, int square) {
    Bitboard bb = square_bb(static_cast<Square>(square));
    pieces_[static_cast<int>(color)][static_cast<int>(type)] &= ~bb;
    occupancies_[static_cast<int>(color)] &= ~bb;
    occupancy_all_ &= ~bb;
    mailbox_[square] = kEmptySquare;
    zobrist_key_ ^= Zobrist::piece_key(color, type, square);
}

PieceType Board::piece_from_char(char c) const {
    switch (std::tolower(static_cast<unsigned char>(c))) {
        case 'p':
            return PieceType::Pawn;
        case 'n':
            return PieceType::Knight;
        case 'b':
            return PieceType::Bishop;
        case 'r':
            return PieceType::Rook;
        case 'q':
            return PieceType::Queen;
        case 'k':
            return PieceType::King;
        default:
            return PieceType::None;
    }
}

PieceType Board::piece_type_at(int square) const {
    if (square < 0 || square >= kBoardSize) {
        return PieceType::None;
    }
    std::uint8_t code = mailbox_[square];
    if (code == kEmptySquare) {
        return PieceType::None;
    }
    return decode_piece_type(code);
}

std::optional<Color> Board::color_at(int square) const {
    if (square < 0 || square >= kBoardSize) {
        return std::nullopt;
    }
    std::uint8_t code = mailbox_[square];
    if (code == kEmptySquare) {
        return std::nullopt;
    }
    return decode_piece_color(code);
}

void Board::set_start_position() {
    set_from_fen(kStartFEN);
}

void Board::set_from_fen(const std::string& fen) {
    clear();
    Zobrist::init();

    std::istringstream iss(fen);
    std::string placement, active, castling, en_passant;
    int halfmove = 0;
    int fullmove = 1;

    if (!(iss >> placement >> active >> castling >> en_passant)) {
        throw std::runtime_error("Invalid FEN string: missing fields");
    }
    if (!(iss >> halfmove)) {
        halfmove = 0;
    }
    if (!(iss >> fullmove)) {
        fullmove = 1;
    }

    int square = 56;  // start from rank 8, file A
    for (char c : placement) {
        if (c == '/') {
            square -= 16;  // move down one rank
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            square += c - '0';
            continue;
        }
        PieceType type = piece_from_char(c);
        if (type == PieceType::None) {
            throw std::runtime_error("Invalid piece character in FEN");
        }
        Color color = std::isupper(static_cast<unsigned char>(c)) ? Color::White : Color::Black;
        place_piece(color, type, square);
        ++square;
    }

    side_to_move_ = (active == "b") ? Color::Black : Color::White;
    if (side_to_move_ == Color::Black) {
        zobrist_key_ ^= Zobrist::side_key();
    }

    std::uint8_t rights = 0;
    if (castling.find('K') != std::string::npos) rights |= kWhiteKingCastle;
    if (castling.find('Q') != std::string::npos) rights |= kWhiteQueenCastle;
    if (castling.find('k') != std::string::npos) rights |= kBlackKingCastle;
    if (castling.find('q') != std::string::npos) rights |= kBlackQueenCastle;
    castling_rights_ = rights;
    zobrist_key_ ^= Zobrist::castling_key(castling_rights_);

    if (en_passant != "-") {
        if (en_passant.size() != 2) {
            throw std::runtime_error("Invalid en passant square in FEN");
        }
        int ep_square = square_from_file_rank(en_passant[0], en_passant[1]);
        en_passant_square_ = ep_square;
        zobrist_key_ ^= Zobrist::en_passant_key(file_of(static_cast<Square>(ep_square)));
    }

    halfmove_clock_ = halfmove;
    fullmove_number_ = fullmove;
}

bool Board::is_square_attacked(Square sq, Color by) const {
    int square = static_cast<int>(sq);
    Bitboard pawns = pieces(by, PieceType::Pawn);
    if (pawn_attacks(opposite_color(by), square) & pawns) {
        return true;
    }
    Bitboard knights = pieces(by, PieceType::Knight);
    if (knight_attacks(square) & knights) {
        return true;
    }
    Bitboard kings = pieces(by, PieceType::King);
    if (king_attacks(square) & kings) {
        return true;
    }
    Bitboard bishops = pieces(by, PieceType::Bishop) | pieces(by, PieceType::Queen);
    if (bishop_attacks(square, occupancy_all_) & bishops) {
        return true;
    }
    Bitboard rooks = pieces(by, PieceType::Rook) | pieces(by, PieceType::Queen);
    if (rook_attacks(square, occupancy_all_) & rooks) {
        return true;
    }
    return false;
}

bool Board::in_check(Color color) const {
    Bitboard king_bb = pieces(color, PieceType::King);
    if (king_bb == 0ULL) {
        return false;
    }
    int king_square = std::countr_zero(king_bb);
    return is_square_attacked(static_cast<Square>(king_square), opposite_color(color));
}

void Board::make_move(const Move& move, State& out_state) {
    out_state.castling_rights = castling_rights_;
    out_state.en_passant_square = en_passant_square_;
    out_state.halfmove_clock = halfmove_clock_;
    out_state.zobrist_key = zobrist_key_;
    out_state.captured_piece = PieceType::None;
    out_state.fullmove_number = fullmove_number_;

    Color us = side_to_move_;
    Color them = opposite_color(us);

    PieceType moving_piece = piece_type_at(move.from);
    if (moving_piece == PieceType::None) {
        throw std::runtime_error("Attempted to move a piece from an empty square");
    }

    if (en_passant_square_ != -1) {
        zobrist_key_ ^= Zobrist::en_passant_key(file_of(static_cast<Square>(en_passant_square_)));
    }
    en_passant_square_ = -1;

    zobrist_key_ ^= Zobrist::castling_key(castling_rights_);

    remove_piece(us, moving_piece, move.from);

    PieceType captured = PieceType::None;
    if (move.is_en_passant()) {
        int cap_sq = move.to + (us == Color::White ? -8 : 8);
        captured = PieceType::Pawn;
        remove_piece(them, captured, cap_sq);
    } else if (move.is_capture()) {
        captured = piece_type_at(move.to);
        if (captured == PieceType::None) {
            throw std::runtime_error("Capture move without a target piece");
        }
        remove_piece(them, captured, move.to);
    }

    PieceType placed_piece = moving_piece;
    if (move.is_promotion()) {
        placed_piece = move.promotion;
    }

    place_piece(us, placed_piece, move.to);

    if (move.is_castle()) {
        if (move.flags & MoveFlag::KingCastle) {
            int rook_from = (us == Color::White) ? static_cast<int>(Square::H1) : static_cast<int>(Square::H8);
            int rook_to = (us == Color::White) ? static_cast<int>(Square::F1) : static_cast<int>(Square::F8);
            remove_piece(us, PieceType::Rook, rook_from);
            place_piece(us, PieceType::Rook, rook_to);
        } else {
            int rook_from = (us == Color::White) ? static_cast<int>(Square::A1) : static_cast<int>(Square::A8);
            int rook_to = (us == Color::White) ? static_cast<int>(Square::D1) : static_cast<int>(Square::D8);
            remove_piece(us, PieceType::Rook, rook_from);
            place_piece(us, PieceType::Rook, rook_to);
        }
    }

    if (moving_piece == PieceType::King) {
        if (us == Color::White) {
            castling_rights_ &= ~(kWhiteKingCastle | kWhiteQueenCastle);
        } else {
            castling_rights_ &= ~(kBlackKingCastle | kBlackQueenCastle);
        }
    } else if (moving_piece == PieceType::Rook) {
        if (us == Color::White) {
            if (move.from == static_cast<int>(Square::A1)) {
                castling_rights_ &= ~kWhiteQueenCastle;
            } else if (move.from == static_cast<int>(Square::H1)) {
                castling_rights_ &= ~kWhiteKingCastle;
            }
        } else {
            if (move.from == static_cast<int>(Square::A8)) {
                castling_rights_ &= ~kBlackQueenCastle;
            } else if (move.from == static_cast<int>(Square::H8)) {
                castling_rights_ &= ~kBlackKingCastle;
            }
        }
    }

    if (captured != PieceType::None) {
        out_state.captured_piece = captured;
        if (!move.is_en_passant()) {
            if (move.to == static_cast<int>(Square::A1)) castling_rights_ &= ~kWhiteQueenCastle;
            if (move.to == static_cast<int>(Square::H1)) castling_rights_ &= ~kWhiteKingCastle;
            if (move.to == static_cast<int>(Square::A8)) castling_rights_ &= ~kBlackQueenCastle;
            if (move.to == static_cast<int>(Square::H8)) castling_rights_ &= ~kBlackKingCastle;
        }
    }

    if (moving_piece == PieceType::Pawn) {
        halfmove_clock_ = 0;
        if (move.is_double_pawn_push()) {
            en_passant_square_ = (move.from + move.to) / 2;
        }
    } else {
        if (captured != PieceType::None) {
            halfmove_clock_ = 0;
        } else {
            ++halfmove_clock_;
        }
    }

    if (en_passant_square_ != -1) {
        zobrist_key_ ^= Zobrist::en_passant_key(file_of(static_cast<Square>(en_passant_square_)));
    }

    zobrist_key_ ^= Zobrist::castling_key(castling_rights_);

    side_to_move_ = them;
    zobrist_key_ ^= Zobrist::side_key();

    if (us == Color::Black) {
        ++fullmove_number_;
    }
}

void Board::undo_move(const Move& move, const State& state) {
    Color them = side_to_move_;
    Color us = opposite_color(them);

    side_to_move_ = us;

    if (en_passant_square_ != -1) {
        zobrist_key_ ^= Zobrist::en_passant_key(file_of(static_cast<Square>(en_passant_square_)));
    }
    zobrist_key_ ^= Zobrist::castling_key(castling_rights_);

    PieceType moved_piece = piece_type_at(move.to);
    remove_piece(us, moved_piece, move.to);

    PieceType original_piece = move.is_promotion() ? PieceType::Pawn : moved_piece;
    place_piece(us, original_piece, move.from);

    if (move.is_castle()) {
        if (move.flags & MoveFlag::KingCastle) {
            int rook_from = (us == Color::White) ? static_cast<int>(Square::F1) : static_cast<int>(Square::F8);
            int rook_to = (us == Color::White) ? static_cast<int>(Square::H1) : static_cast<int>(Square::H8);
            remove_piece(us, PieceType::Rook, rook_from);
            place_piece(us, PieceType::Rook, rook_to);
        } else {
            int rook_from = (us == Color::White) ? static_cast<int>(Square::D1) : static_cast<int>(Square::D8);
            int rook_to = (us == Color::White) ? static_cast<int>(Square::A1) : static_cast<int>(Square::A8);
            remove_piece(us, PieceType::Rook, rook_from);
            place_piece(us, PieceType::Rook, rook_to);
        }
    }

    if (state.captured_piece != PieceType::None) {
        if (move.is_en_passant()) {
            int cap_sq = move.to + (us == Color::White ? -8 : 8);
            place_piece(them, PieceType::Pawn, cap_sq);
        } else {
            place_piece(them, state.captured_piece, move.to);
        }
    }

    castling_rights_ = state.castling_rights;
    en_passant_square_ = state.en_passant_square;
    halfmove_clock_ = state.halfmove_clock;
    zobrist_key_ = state.zobrist_key;
    fullmove_number_ = state.fullmove_number;
}

void Board::make_null_move(State& out_state) {
    out_state.castling_rights = castling_rights_;
    out_state.en_passant_square = en_passant_square_;
    out_state.halfmove_clock = halfmove_clock_;
    out_state.zobrist_key = zobrist_key_;
    out_state.captured_piece = PieceType::None;
    out_state.fullmove_number = fullmove_number_;

    if (en_passant_square_ != -1) {
        zobrist_key_ ^= Zobrist::en_passant_key(file_of(static_cast<Square>(en_passant_square_)));
    }
    en_passant_square_ = -1;

    Color us = side_to_move_;
    side_to_move_ = opposite_color(us);
    zobrist_key_ ^= Zobrist::side_key();

    if (us == Color::Black) {
        ++fullmove_number_;
    }

    ++halfmove_clock_;
}

void Board::undo_null_move(const State& state) {
    side_to_move_ = opposite_color(side_to_move_);
    castling_rights_ = state.castling_rights;
    en_passant_square_ = state.en_passant_square;
    halfmove_clock_ = state.halfmove_clock;
    zobrist_key_ = state.zobrist_key;
    fullmove_number_ = state.fullmove_number;
}

std::string Board::fen() const {
    std::ostringstream oss;
    for (int rank = 7; rank >= 0; --rank) {
        int empty_count = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            PieceType type = piece_type_at(sq);
            if (type == PieceType::None) {
                ++empty_count;
            } else {
                if (empty_count > 0) {
                    oss << empty_count;
                    empty_count = 0;
                }
                auto color = color_at(sq);
                char piece_char = '1';
                switch (type) {
                    case PieceType::Pawn:
                        piece_char = 'p';
                        break;
                    case PieceType::Knight:
                        piece_char = 'n';
                        break;
                    case PieceType::Bishop:
                        piece_char = 'b';
                        break;
                    case PieceType::Rook:
                        piece_char = 'r';
                        break;
                    case PieceType::Queen:
                        piece_char = 'q';
                        break;
                    case PieceType::King:
                        piece_char = 'k';
                        break;
                    default:
                        break;
                }
                if (color && *color == Color::White) {
                    piece_char = std::toupper(piece_char);
                }
                oss << piece_char;
            }
        }
        if (empty_count > 0) {
            oss << empty_count;
        }
        if (rank > 0) {
            oss << '/';
        }
    }

    oss << ' ' << (side_to_move_ == Color::White ? 'w' : 'b') << ' ';

    std::string castling_str;
    if (castling_rights_ & kWhiteKingCastle) castling_str += 'K';
    if (castling_rights_ & kWhiteQueenCastle) castling_str += 'Q';
    if (castling_rights_ & kBlackKingCastle) castling_str += 'k';
    if (castling_rights_ & kBlackQueenCastle) castling_str += 'q';
    if (castling_str.empty()) castling_str = "-";
    oss << castling_str << ' ';

    if (en_passant_square_ != -1) {
        oss << square_to_string(static_cast<Square>(en_passant_square_));
    } else {
        oss << '-';
    }

    oss << ' ' << halfmove_clock_ << ' ' << fullmove_number_;
    return oss.str();
}

}  // namespace chiron

