#pragma once
// Phase 3B-2 of the lc0-split migration: the C++ self-play CLIENT's HTTP surface
// (cpp-httplib, header-only, vendored at cpp/third_party/httplib.h — the
// user-confirmed lc0-spirit / portable-volunteer choice). Speaks the EXISTING
// fleet contract (no protocol change), so a native client can fetch the net by
// content address and upload a ccz chunk with zero Python.
//
// This slice (3B-2a) is the two BINARY operations, where wire-format parity with
// fleet_server is the risk:
//   - GET  /get_network?sha=<sha>   -> the net bytes (content-addressed)
//   - POST /upload_game (multipart) -> lands the ccz chunk in the server's buffer/
// The job-JSON next_game parse + the full self-play loop + the standalone
// executable are 3B-2b.
//
// Plain HTTP only (the fleet uses http://); CPPHTTPLIB_OPENSSL_SUPPORT is NOT
// defined, so no OpenSSL dependency.
#include <string>
#include <utility>

#include "httplib.h"

namespace cc {

// GET <base_url><path>. Returns (status, body); (0, "") on transport failure so
// the caller can retry — mirrors how the Python client tolerates a flaky LAN.
inline std::pair<int, std::string> http_get(const std::string& base_url, const std::string& path) {
    httplib::Client cli(base_url.c_str());
    cli.set_keep_alive(false);
    auto res = cli.Get(path.c_str());
    if (!res) return {0, std::string()};
    return {res->status, res->body};
}

// Content-addressed net fetch: GET /get_network?sha=<sha>. (status, bytes).
inline std::pair<int, std::string> fleet_get_network(const std::string& base_url,
                                                     const std::string& sha) {
    return http_get(base_url, "/get_network?sha=" + sha);
}

// lc0-canonical multipart game upload: POST /upload_game with parts
//   filename     (value-only: the NNN_*.pkl name the server validates + writes under)
//   trainingdata (the ccz chunk bytes — gzipped JSON; std::string is byte-safe)
//   meta         (optional .pkl.meta JSON)
// Matches fleet_server._parse_multipart (which ignores per-part Content-Type and
// reads the content bytes, so cpp-httplib's framing parses cleanly). Returns
// (status, body) — 200/"ok" on success.
inline std::pair<int, std::string> fleet_upload_game(const std::string& base_url,
                                                     const std::string& filename,
                                                     const std::string& chunk,
                                                     const std::string& meta) {
    httplib::Client cli(base_url.c_str());
    cli.set_keep_alive(false);
    httplib::MultipartFormDataItems items = {
        {"filename", filename, "", ""},
        {"trainingdata", chunk, "game.pkl", "application/octet-stream"},
    };
    if (!meta.empty()) items.push_back({"meta", meta, "game.pkl.meta", ""});
    auto res = cli.Post("/upload_game", items);
    if (!res) return {0, std::string()};
    return {res->status, res->body};
}

// POST /next_game — claim a job. Returns (status, raw job JSON). The C++-side JSON
// parse (type/sha/bin_sha/params) is 3B-2b; for now the body is returned verbatim.
inline std::pair<int, std::string> fleet_next_game(const std::string& base_url) {
    httplib::Client cli(base_url.c_str());
    cli.set_keep_alive(false);
    auto res = cli.Post("/next_game", "", "application/x-www-form-urlencoded");
    if (!res) return {0, std::string()};
    return {res->status, res->body};
}

// POST /match_result — report one client-played GATE outcome (Phase 4b). `json_body`
// is the result JSON ({match_id, seed, opp, cand_white, outcome}); the server tallies
// it into match_results/ for the arena (200/"ok", or "stale" for a closed gate).
inline std::pair<int, std::string> fleet_post_match_result(const std::string& base_url,
                                                           const std::string& json_body) {
    httplib::Client cli(base_url.c_str());
    cli.set_keep_alive(false);
    auto res = cli.Post("/match_result", json_body, "application/json");
    if (!res) return {0, std::string()};
    return {res->status, res->body};
}

}  // namespace cc
