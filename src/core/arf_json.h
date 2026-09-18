#pragma once
// arf_json.h — minimal, dependency-free JSON *writer* for ARF documents.
//
// arf_writer.cpp only ever WRITES arf.json (an ARF container is produced
// here, never consumed), so a full JSON library is unnecessary. This is a
// small order-preserving value tree + RFC 8259 serializer covering exactly
// what building an ARF document needs (see ARF.md and ISO/IEC 23090-39,
// overview: https://ieeexplore.ieee.org/document/11667221).

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace arf_json
{

class Value
{
public:
    Value() : kind_(Kind::Null) {}
    Value(std::nullptr_t) : kind_(Kind::Null) {}
    Value(bool b) : kind_(Kind::Bool), bool_(b) {}
    Value(int v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(unsigned v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(long v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(unsigned long v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(long long v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(double v) : kind_(Kind::Number), num_(v) {}
    Value(float v) : kind_(Kind::Number), num_(static_cast<double>(v)) {}
    Value(const char* s) : kind_(Kind::String), str_(s) {}
    Value(const std::string& s) : kind_(Kind::String), str_(s) {}
    Value(std::string&& s) : kind_(Kind::String), str_(std::move(s)) {}

    static Value object() { Value v; v.kind_ = Kind::Object; return v; }
    static Value array()  { Value v; v.kind_ = Kind::Array;  return v; }

    // Object: insert/replace `key` (kind must already be Object, or Null —
    // the first set()/push_back() call establishes the kind). Returns *this
    // for chaining: obj.set("a", 1).set("b", 2).
    Value& set(const std::string& key, Value v);

    // Array: append (kind must already be Array, or Null).
    Value& push_back(Value v);

    bool is_null() const { return kind_ == Kind::Null; }

    // Serialize. `indent` spaces per nesting level; 0 = compact (no
    // whitespace). Not reentrant-safe across threads on the same Value, but
    // arf_writer.cpp builds one document per person, single-threaded.
    std::string dump(int indent = 2) const;

private:
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind_;
    bool   bool_ = false;
    double num_  = 0.0;
    std::string str_;
    std::vector<Value> arr_;
    std::vector<std::pair<std::string, Value>> obj_;   // insertion-ordered

    void write(std::string& out, int indent, int depth) const;
};

// RFC 8259 string escaping (quotes, backslash, control characters).
// Exposed separately because arf_writer.cpp also needs it for hand-rolled
// fragments (e.g. writing large numeric arrays without going through Value,
// which would be wasteful for an 18439-vertex mesh).
std::string escape(const std::string& s);

} // namespace arf_json
