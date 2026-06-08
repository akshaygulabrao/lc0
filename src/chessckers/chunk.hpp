#pragma once
// Phase 3A of the lc0-split migration: C++-side training-chunk encoding.
//
// Mirrors chessckers_engine.training_chunk.encode_chunk + selfplay_az.
// az_game_to_examples — turns a PureGame (from selfplay.hpp) into the SAME
// gzipped-JSON "ccz1" chunk the Python self-play path produces, so a native
// self-play client can encode+upload without any Python in the loop.
//
// DATA-ONLY by construction (gzipped UTF-8 JSON, never pickle): an untrusted
// volunteer / LAN self-play client uploads these to the trainer, which decodes
// with json.loads(gzip.decompress(...)) — no code execution. See the security
// rationale in training_chunk.py.
//
// The gate is TENSOR-identical (not byte-identical) round-trip: this chunk
// decoded by Python's decode_chunk yields AZExamples that encode to the same
// position/move/target tensors as az_game_to_examples of the equivalent game.
// Floats are emitted at 17 significant digits so json.loads recovers the exact
// same IEEE-754 double C++ computed (v/total is deterministic across both).
#include <cstdio>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include <zlib.h>

#include "movegen.hpp"
#include "movegen_white.hpp"
#include "native_move.hpp"
#include "selfplay.hpp"

namespace cc {

// ---- minimal JSON emitters (append to a std::string) ----

inline void json_escape(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

// 17 sig digits round-trips any double exactly; ensure the token reads back as a
// JSON float (append ".0" when %.17g produced a bare integer like "0" or "1").
inline void json_double(std::string& out, double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    std::string s(buf);
    if (s.find_first_of(".eEnN") == std::string::npos) s += ".0";
    out += s;
}

// Object builder that tracks comma placement; values that are absent are emitted
// as JSON null so the decoded dict has EXACTLY the keys the Python dict builders
// produce (key-for-key parity with bindings.cpp native_move_to_dict).
struct JsonObj {
    std::string& o;
    bool first = true;
    explicit JsonObj(std::string& out) : o(out) { o += '{'; }
    void key(const char* k) {
        if (!first) o += ',';
        first = false;
        o += '"';
        o += k;
        o += "\":";
    }
    void s(const char* k, const std::string& v) { key(k); json_escape(o, v); }
    void s(const char* k, const char* v) { key(k); std::string t(v); json_escape(o, t); }
    void null(const char* k) { key(k); o += "null"; }
    void i(const char* k, long v) { key(k); o += std::to_string(v); }
    void b(const char* k, bool v) { key(k); o += (v ? "true" : "false"); }
    void d(const char* k, double v) { key(k); json_double(o, v); }
    void str_arr(const char* k, const std::vector<std::string>& v) {
        key(k);
        o += '[';
        for (size_t j = 0; j < v.size(); ++j) {
            if (j) o += ',';
            json_escape(o, v[j]);
        }
        o += ']';
    }
    void int_arr(const char* k, const std::vector<int>& v) {
        key(k);
        o += '[';
        for (size_t j = 0; j < v.size(); ++j) {
            if (j) o += ',';
            o += std::to_string(v[j]);
        }
        o += ']';
    }
    void end() { o += '}'; }
};

// ---- per-move-type JSON (verbatim mirrors of the bindings.cpp dict builders) ----

inline void white_move_json(std::string& o, const WCandidate& c) {
    JsonObj j(o);
    j.s("uci", white_uci(c));
    j.s("from", square_name(c.from_sq));
    j.s("to", square_name(c.to_sq));
    j.s("piece", wpiece_name(c.piece));
    j.s("color", "white");
    if (c.capture_sq >= 0) j.s("capture", square_name(c.capture_sq));
    else j.null("capture");
    j.null("waypoints");
    j.null("chainHops");
    if (c.promotion) j.s("promotion", *c.promotion);
    else j.null("promotion");
    j.null("demotedKings");
    j.null("demotionsRequired");
    j.null("sourceKingPositions");
    j.null("deployCount");
    j.end();
}

inline void white_castling_alt_json(std::string& o, const WCandidate& c) {
    JsonObj j(o);
    j.s("uci", square_name(c.from_sq) + square_name(c.castling_rook_sq));
    j.s("from", square_name(c.from_sq));
    j.s("to", square_name(c.castling_rook_sq));
    j.s("piece", "king");
    j.s("color", "white");
    j.null("capture");
    j.null("waypoints");
    j.null("chainHops");
    j.null("promotion");
    j.null("demotedKings");
    j.null("demotionsRequired");
    j.null("sourceKingPositions");
    j.null("deployCount");
    j.end();
}

inline void simple_move_json(std::string& o, const std::string& uci, const std::string& from,
                             const std::string& to, const std::string& piece, bool has_deploy,
                             int deploy_count) {
    JsonObj j(o);
    j.s("uci", uci);
    j.s("from", from);
    j.s("to", to);
    j.s("piece", piece);
    j.s("color", "black");
    j.null("capture");
    j.null("waypoints");
    j.null("chainHops");
    j.null("promotion");
    j.null("demotedKings");
    j.null("demotionsRequired");
    j.null("sourceKingPositions");
    if (has_deploy) j.i("deployCount", deploy_count);
    else j.null("deployCount");
    j.end();
}

inline void charge_json(std::string& o, const ChargeMove& c) {
    JsonObj j(o);
    j.s("uci", c.uci);
    j.s("from", c.from_name);
    j.s("to", c.to_name);
    j.s("piece", c.piece);
    j.s("color", "black");
    if (c.capture) j.s("capture", *c.capture);
    else j.null("capture");
    if (c.waypoints) j.str_arr("waypoints", *c.waypoints);
    else j.null("waypoints");
    j.null("chainHops");
    j.null("promotion");
    if (c.demoted_kings) j.int_arr("demotedKings", *c.demoted_kings);
    else j.null("demotedKings");
    if (c.demotions_required) j.i("demotionsRequired", *c.demotions_required);
    else j.null("demotionsRequired");
    if (c.source_king_positions) j.int_arr("sourceKingPositions", *c.source_king_positions);
    else j.null("sourceKingPositions");
    j.null("deployCount");
    j.end();
}

inline void chain_json(std::string& o, const ChainMove& m) {
    JsonObj j(o);
    j.s("uci", m.uci);
    j.s("from", m.from_name);
    j.s("to", m.to_name);
    j.s("piece", m.piece);
    j.s("color", "black");
    if (m.capture) j.s("capture", *m.capture);
    else j.null("capture");
    if (m.waypoints) j.str_arr("waypoints", *m.waypoints);
    else j.null("waypoints");
    j.str_arr("chainHops", m.chain_hops);
    j.null("promotion");
    j.null("demotedKings");
    j.null("demotionsRequired");
    j.null("sourceKingPositions");
    j.null("deployCount");
    j.str_arr("_chain_all_captures", m.chain_all_captures);
    j.i("cadence", m.cadence);
    j.b("_is_suicide", m.is_suicide);
    j.b("_chain_promotes", m.chain_promotes);
    j.end();
}

inline void native_move_json(std::string& o, const NativeMove& m) {
    if (m.is_white) {
        if (m.is_castling_alt) white_castling_alt_json(o, m.white_src);
        else white_move_json(o, m.white_src);
        return;
    }
    std::visit(
        [&](auto&& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, QuietMove>)
                simple_move_json(o, x.uci, x.from_name, x.to_name, x.piece, false, 0);
            else if constexpr (std::is_same_v<T, DeployMove>)
                simple_move_json(o, x.uci, x.from_name, x.to_name, x.piece, true, x.deploy_count);
            else if constexpr (std::is_same_v<T, ChargeMove>)
                charge_json(o, x);
            else
                chain_json(o, x);
        },
        m.black_src);
}

// ---- gzip via zlib (windowBits 31 == gzip wrapper, mtime defaults to 0) ----

inline std::string gzip_compress(const std::string& raw, int level = 6) {
    z_stream zs{};
    if (deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("deflateInit2 failed");
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(raw.data()));
    zs.avail_in = static_cast<uInt>(raw.size());
    std::string out;
    char buf[32768];
    int ret;
    do {
        zs.next_out = reinterpret_cast<Bytef*>(buf);
        zs.avail_out = sizeof(buf);
        ret = deflate(&zs, Z_FINISH);
        out.append(buf, sizeof(buf) - zs.avail_out);
    } while (ret == Z_OK);
    deflateEnd(&zs);
    if (ret != Z_STREAM_END) throw std::runtime_error("deflate failed");
    return out;
}

// ---- PureGame -> ccz1 chunk (gzipped JSON) ----
// Mirrors selfplay_az.az_game_to_examples: WDL one-hot from outcome+side POV,
// visit_distribution = visits/total, moves_left_target = plies-to-end (n - i).
inline std::string encode_chunk(const PureGame& game) {
    std::vector<double> wdl_white, wdl_black;
    if (game.outcome == "draw") {
        wdl_white = {0.0, 1.0, 0.0};
        wdl_black = {0.0, 1.0, 0.0};
    } else if (game.outcome == "white") {
        wdl_white = {1.0, 0.0, 0.0};
        wdl_black = {0.0, 0.0, 1.0};
    } else {  // black
        wdl_white = {0.0, 0.0, 1.0};
        wdl_black = {1.0, 0.0, 0.0};
    }

    std::string o = "{\"schema\":\"ccz1\",\"examples\":[";
    const int n = static_cast<int>(game.records.size());
    for (int i = 0; i < n; ++i) {
        const PureRecord& rec = game.records[i];
        if (i) o += ',';
        JsonObj j(o);
        j.s("fen", rec.fen);
        j.key("legal_moves");
        o += '[';
        for (size_t k = 0; k < rec.legal.size(); ++k) {
            if (k) o += ',';
            native_move_json(o, rec.legal[k]);
        }
        o += ']';
        long total = 0;
        for (int v : rec.visits) total += v;
        if (total == 0) total = 1;
        j.key("visit_distribution");
        o += '[';
        for (size_t k = 0; k < rec.visits.size(); ++k) {
            if (k) o += ',';
            json_double(o, static_cast<double>(rec.visits[k]) / static_cast<double>(total));
        }
        o += ']';
        const std::vector<double>& wdl = rec.side_white ? wdl_white : wdl_black;
        j.key("wdl_target");
        o += '[';
        for (size_t k = 0; k < wdl.size(); ++k) {
            if (k) o += ',';
            json_double(o, wdl[k]);
        }
        o += ']';
        j.d("moves_left_target", static_cast<double>(n - i));
        j.end();
    }
    o += "]}";
    return gzip_compress(o);
}

}  // namespace cc
