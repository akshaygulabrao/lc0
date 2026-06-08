#pragma once
// Minimal recursive-descent JSON parser for Phase 3B-2b — just enough to read the
// fleet server's next_game job ({"type","sha","bin_sha","params":{...}}) in the
// standalone C++ client. Parsing (not emitting — chunk.hpp emits); the client
// never trusts the bytes as code, so a hand-rolled value parser is the lc0-spirit
// minimal dependency (no second vendored lib beyond cpp-httplib). Throws
// std::runtime_error on malformed input.
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc {

struct JsonValue {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    bool is_obj() const { return type == Obj; }
    const JsonValue* find(const std::string& k) const {
        if (type != Obj) return nullptr;
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string get_str(const std::string& k, const std::string& dflt = "") const {
        const JsonValue* v = find(k);
        return (v && v->type == Str) ? v->str : dflt;
    }
    double get_num(const std::string& k, double dflt) const {
        const JsonValue* v = find(k);
        return (v && v->type == Num) ? v->num : dflt;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : p_(s.data()), end_(s.data() + s.size()) {}
    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        return v;
    }

private:
    const char* p_;
    const char* end_;

    [[noreturn]] void fail(const char* msg) { throw std::runtime_error(std::string("json: ") + msg); }
    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    char peek() {
        if (p_ >= end_) fail("unexpected eof");
        return *p_;
    }
    void expect_lit(const char* lit) {
        for (const char* q = lit; *q; ++q) {
            if (p_ >= end_ || *p_ != *q) fail("bad literal");
            ++p_;
        }
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Str;
            v.str = parse_string();
            return v;
        }
        if (c == 't') {
            expect_lit("true");
            JsonValue v;
            v.type = JsonValue::Bool;
            v.b = true;
            return v;
        }
        if (c == 'f') {
            expect_lit("false");
            JsonValue v;
            v.type = JsonValue::Bool;
            v.b = false;
            return v;
        }
        if (c == 'n') {
            expect_lit("null");
            return JsonValue{};
        }
        return parse_number();
    }

    std::string parse_string() {
        if (peek() != '"') fail("expected string");
        ++p_;
        std::string out;
        while (p_ < end_) {
            char c = *p_++;
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p_ >= end_) fail("bad escape");
            char e = *p_++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (end_ - p_ < 4) fail("bad \\u");
                    int cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        char h = *p_++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= h - '0';
                        else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        else fail("bad hex");
                    }
                    if (cp < 0x80) {
                        out += (char)cp;
                    } else if (cp < 0x800) {
                        out += (char)(0xC0 | (cp >> 6));
                        out += (char)(0x80 | (cp & 0x3F));
                    } else {
                        out += (char)(0xE0 | (cp >> 12));
                        out += (char)(0x80 | ((cp >> 6) & 0x3F));
                        out += (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: fail("bad escape char");
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_number() {
        const char* start = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        while (p_ < end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' ||
                             *p_ == 'E' || *p_ == '+' || *p_ == '-'))
            ++p_;
        if (p_ == start) fail("bad number");
        JsonValue v;
        v.type = JsonValue::Num;
        v.num = std::strtod(std::string(start, p_).c_str(), nullptr);
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Arr;
        ++p_;  // [
        skip_ws();
        if (peek() == ']') {
            ++p_;
            return v;
        }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++p_;
                continue;
            }
            if (c == ']') {
                ++p_;
                break;
            }
            fail("expected , or ]");
        }
        return v;
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Obj;
        ++p_;  // {
        skip_ws();
        if (peek() == '}') {
            ++p_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected :");
            ++p_;
            v.obj[key] = parse_value();
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++p_;
                continue;
            }
            if (c == '}') {
                ++p_;
                break;
            }
            fail("expected , or }");
        }
        return v;
    }
};

inline JsonValue parse_json(const std::string& s) { return JsonParser(s).parse(); }

}  // namespace cc
