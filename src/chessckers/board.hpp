// Chessckers C++ engine — Slice 0: board representation + FEN round-trip.
//
// Mirrors the PyVariant `State` (chessckers_engine/variant_py/state.py) and the
// bb-decomposition the Rust crate (rust/chessckers_movegen) operates on, so that
// later slices expose the same call surface and reuse the existing parity tests
// as oracles. This slice implements ONLY parse_fen / serialize_fen; it must match
// `serialize_fen(parse_fen(fen))` byte-for-byte over the canonical golden corpus.
//
// Black towers are encoded onto the bitboards exactly as PyVariant does: a
// Stone-top tower is a black pawn ('p'), a King-top tower is a black king ('k').
// FEN parsing here is purely syntactic (no chess-legality validation) — python
// -chess's Board constructor is likewise syntactic, which is why pawns-on-rank-8
// (encoded stones) round-trip cleanly.
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc {

// python-chess castling_rights bitmask: bits are the ROOK home squares.
// a1=0 (Q), h1=7 (K), a8=56 (q), h8=63 (k). Matches state.py::_castling_field.
constexpr uint64_t BB_A1 = 1ULL << 0;
constexpr uint64_t BB_H1 = 1ULL << 7;
constexpr uint64_t BB_A8 = 1ULL << 56;
constexpr uint64_t BB_H8 = 1ULL << 63;

constexpr uint64_t BB_RANK_1 = 0xFFULL;
constexpr uint64_t BB_RANK_8 = 0xFFULL << 56;
constexpr uint64_t BB_FILE_A = 0x0101010101010101ULL;
constexpr uint64_t BB_FILE_H = 0x8080808080808080ULL;

inline int lsb(uint64_t bb) { return bb ? __builtin_ctzll(bb) : -1; }
inline int msb(uint64_t bb) { return bb ? 63 - __builtin_clzll(bb) : -1; }

struct Board {
    // piece-type bitboards (color-agnostic); bit i == square i (a1=0 .. h8=63).
    uint64_t pawns = 0, knights = 0, bishops = 0, rooks = 0, queens = 0, kings = 0;
    uint64_t occupied_white = 0, occupied_black = 0;
    uint64_t castling_rights = 0;  // python-chess style (rook-corner bits)
    int ep_square = -1;            // -1 == none
    bool turn_white = true;        // true == 'w'
    int halfmove = 0;
    int fullmove = 1;
    // Chessckers turn/win state (the FEN's trailing {wm,r8} block). Mirrors
    // variant_py.state.State: white_moves_left = White sub-moves left this turn
    // (2 only at the opening double-move, 1 normally); rank8_count = consecutive
    // completed White turns with the king on rank 8 (3 => White wins; a check
    // resets to 0).
    int white_moves_left = 1;
    int rank8_count = 0;
    // ordered ascending by square -> matches sorted(stacks.items()) on serialize.
    std::map<uint8_t, std::string> stacks;

    uint64_t occupied() const { return occupied_white | occupied_black; }
};

// --- small string helpers -------------------------------------------------

inline std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && std::isspace((unsigned char)s[i])) ++i;
        size_t start = i;
        while (i < n && !std::isspace((unsigned char)s[i])) ++i;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

// --- square helpers (chess.parse_square / chess.square_name) ---------------

inline int parse_square(const std::string& n) {
    if (n.size() != 2) throw std::invalid_argument("invalid square: " + n);
    return (n[1] - '1') * 8 + (n[0] - 'a');
}

inline std::string square_name(int sq) {
    std::string s;
    s += char('a' + sq % 8);
    s += char('1' + sq / 8);
    return s;
}

// --- piece <-> bitboard ----------------------------------------------------

inline void set_piece(Board& b, int sq, char letter) {
    const uint64_t m = 1ULL << sq;
    const bool white = std::isupper((unsigned char)letter) != 0;
    switch (std::tolower((unsigned char)letter)) {
        case 'p': b.pawns |= m; break;
        case 'n': b.knights |= m; break;
        case 'b': b.bishops |= m; break;
        case 'r': b.rooks |= m; break;
        case 'q': b.queens |= m; break;
        case 'k': b.kings |= m; break;
        default: throw std::invalid_argument(std::string("bad piece letter: ") + letter);
    }
    if (white) b.occupied_white |= m; else b.occupied_black |= m;
}

inline char piece_letter(const Board& b, int sq) {
    const uint64_t m = 1ULL << sq;
    char c;
    if (b.pawns & m) c = 'p';
    else if (b.knights & m) c = 'n';
    else if (b.bishops & m) c = 'b';
    else if (b.rooks & m) c = 'r';
    else if (b.queens & m) c = 'q';
    else if (b.kings & m) c = 'k';
    else return 0;
    if (b.occupied_white & m) c = char(std::toupper((unsigned char)c));
    return c;
}

// --- board grid (FEN piece-placement field) --------------------------------

inline void parse_board(Board& b, const std::string& bs) {
    int rank = 7;  // FEN lists rank 8 first -> square indices 56..63 -> rank 7
    int file = 0;
    for (char c : bs) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        set_piece(b, rank * 8 + file, c);
        ++file;
    }
}

inline std::string board_fen(const Board& b) {
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empties = 0;
        for (int file = 0; file < 8; ++file) {
            char c = piece_letter(b, rank * 8 + file);
            if (c == 0) { ++empties; continue; }
            if (empties) { out += char('0' + empties); empties = 0; }
            out += c;
        }
        if (empties) out += char('0' + empties);
        if (rank > 0) out += '/';
    }
    return out;
}

// --- castling -------------------------------------------------------------
// king(color): python-chess returns msb of the color's king bitboard, or None
// (here -1) if absent. Chessckers has multiple Black kings, so this is the
// highest-square Black king — matching python-chess exactly.
inline int king_sq(const Board& b, bool white) {
    const uint64_t km = b.kings & (white ? b.occupied_white : b.occupied_black);
    return msb(km);  // msb(0) == -1 == None
}

// Faithful port of chess.Board._set_castling_fen: each flag is resolved against
// the actual rooks/king on the backrank (NOT a naive corner-bit mapping). This
// reproduces python-chess's quirks for degenerate positions (e.g. STARTING_FEN,
// where Black has no rooks, so 'k' falls back to file-H but 'q' adds nothing).
// Requires the board bitboards to be populated first.
inline uint64_t set_castling_fen(const Board& b, const std::string& f) {
    if (f.empty() || f == "-") return 0;
    uint64_t rights = 0;
    for (char flag : f) {
        const bool white = std::isupper((unsigned char)flag) != 0;
        const char fl = char(std::tolower((unsigned char)flag));
        const uint64_t backrank = white ? BB_RANK_1 : BB_RANK_8;
        const uint64_t rooks = (white ? b.occupied_white : b.occupied_black) & b.rooks & backrank;
        const int king = king_sq(b, white);
        if (fl == 'q') {
            if (king != -1 && lsb(rooks) < king) rights |= rooks & (0ULL - rooks);  // lsb rook
            else rights |= BB_FILE_A & backrank;
        } else if (fl == 'k') {
            const int rook = msb(rooks);
            if (king != -1 && king < rook) rights |= 1ULL << rook;
            else rights |= BB_FILE_H & backrank;
        } else {  // explicit file letter (Chess960 / X-FEN); never in our corpus
            rights |= (BB_FILE_A << (fl - 'a')) & backrank;
        }
    }
    return rights;
}

// state.py::_castling_field — inverse of the above for serialization.
inline std::string castling_field(uint64_t r) {
    if (!r) return "-";
    std::string s;
    if (r & BB_H1) s += 'K';
    if (r & BB_A1) s += 'Q';
    if (r & BB_H8) s += 'k';
    if (r & BB_A8) s += 'q';
    return s.empty() ? "-" : s;
}

// --- FEN parse / serialize -------------------------------------------------

inline Board parse_fen(const std::string& fen_in) {
    // Mirrors _FEN_HEAD_RE: ^([^\s\[]+)(?:\[([^\]]*)\])?(\s.*)?$
    const std::string fen = strip(fen_in);
    Board b;
    size_t i = 0;
    while (i < fen.size() && fen[i] != '[' && !std::isspace((unsigned char)fen[i])) ++i;
    const std::string board_str = fen.substr(0, i);

    bool have_overlay = false;
    std::string overlay_str;
    if (i < fen.size() && fen[i] == '[') {
        const size_t j = fen.find(']', i);
        if (j == std::string::npos) throw std::invalid_argument("unterminated FEN overlay");
        overlay_str = fen.substr(i + 1, j - (i + 1));
        have_overlay = true;
        i = j + 1;
    }
    std::string rest = (i < fen.size()) ? strip(fen.substr(i)) : "";
    if (rest.empty()) rest = "w - - 0 1";  // parse_fen default when rest absent

    // Pull off the optional trailing {wm,r8} block before tokenizing the six
    // standard fields (mirrors state.py _FEN_CKSTATE_RE).
    int ck_wm = 1, ck_r8 = 0;
    {
        const size_t lb = rest.rfind('{');
        if (lb != std::string::npos) {
            const size_t rb = rest.find('}', lb);
            const std::string inside =
                rest.substr(lb + 1, (rb == std::string::npos ? rest.size() : rb) - lb - 1);
            rest = strip(rest.substr(0, lb));
            for (const std::string& kv : split(inside, ',')) {
                const std::string e = strip(kv);
                const size_t c = e.find(':');
                if (e.empty() || c == std::string::npos) continue;
                const std::string key = strip(e.substr(0, c));
                const int val = std::stoi(strip(e.substr(c + 1)));
                if (key == "wm") ck_wm = val;
                else if (key == "r8") ck_r8 = val;
            }
        }
    }

    const std::vector<std::string> t = split_ws(rest);
    const std::string turn = t.size() > 0 ? t[0] : "w";
    const std::string cast = t.size() > 1 ? t[1] : "-";
    const std::string ep   = t.size() > 2 ? t[2] : "-";
    const std::string hm   = t.size() > 3 ? t[3] : "0";
    const std::string fm   = t.size() > 4 ? t[4] : "1";

    parse_board(b, board_str);
    b.turn_white = (turn == "w");
    b.castling_rights = set_castling_fen(b, cast);  // position-aware (needs board parsed)
    b.ep_square = (ep == "-") ? -1 : parse_square(ep);
    b.halfmove = std::stoi(hm);
    b.fullmove = std::stoi(fm);

    if (have_overlay && !overlay_str.empty()) {
        for (const std::string& entry : split(overlay_str, ',')) {
            const std::string e = strip(entry);
            if (e.empty()) continue;
            const size_t c = e.find(':');
            const std::string sqn = (c == std::string::npos) ? e : e.substr(0, c);
            const std::string pieces = (c == std::string::npos) ? "" : e.substr(c + 1);
            b.stacks[(uint8_t)parse_square(sqn)] = pieces;
        }
    }
    b.white_moves_left = ck_wm;
    b.rank8_count = ck_r8;
    return b;
}

inline std::string serialize_fen(const Board& b) {
    const std::string board_part = board_fen(b);
    // En passant is STRUCTURALLY IMPOSSIBLE in Chessckers: only White has FIDE
    // pawns, and Black (Stones, not pawns) can never ep-capture — so python-chess's
    // board.fen() (en_passant="legal") ALWAYS downgrades the ep field to '-' here.
    // Emit '-' unconditionally to stay byte-equal with PyVariant. (The internal
    // ep_square a White double-push sets is vestigial — never legally capturable,
    // never used by move-gen — so its staleness across plies doesn't matter.)
    const std::string ep = "-";
    const std::string rest = std::string(b.turn_white ? "w" : "b") + " " +
                             castling_field(b.castling_rights) + " " + ep + " " +
                             std::to_string(b.halfmove) + " " + std::to_string(b.fullmove);
    // Optional trailing {wm,r8} block — emitted only when non-default, so
    // ordinary positions and every pre-existing FEN serialize unchanged.
    std::string ck;
    if (b.white_moves_left != 1) ck += "wm:" + std::to_string(b.white_moves_left);
    if (b.rank8_count != 0) {
        if (!ck.empty()) ck += ',';
        ck += "r8:" + std::to_string(b.rank8_count);
    }
    const std::string suffix = ck.empty() ? "" : (" {" + ck + "}");

    if (b.stacks.empty()) return board_part + " " + rest + suffix;

    std::string overlay;
    bool first = true;
    for (const auto& [sq, pieces] : b.stacks) {  // std::map -> ascending square
        if (!first) overlay += ',';
        first = false;
        overlay += square_name(sq) + ":" + pieces;
    }
    return board_part + "[" + overlay + "] " + rest + suffix;
}

}  // namespace cc
