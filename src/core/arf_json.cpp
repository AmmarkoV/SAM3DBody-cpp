// arf_json.cpp — see arf_json.h.

#include "arf_json.h"

#include <cmath>
#include <cstdio>

namespace arf_json
{

Value& Value::set(const std::string& key, Value v)
{
    if (kind_ == Kind::Null) kind_ = Kind::Object;
    for (auto& kv : obj_)
        if (kv.first == key) { kv.second = std::move(v); return *this; }
    obj_.emplace_back(key, std::move(v));
    return *this;
}

Value& Value::push_back(Value v)
{
    if (kind_ == Kind::Null) kind_ = Kind::Array;
    arr_.push_back(std::move(v));
    return *this;
}

std::string escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (unsigned char c : s)
    {
        switch (c)
        {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20)
                {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                }
                else out.push_back((char)c);
        }
    }
    out.push_back('"');
    return out;
}

namespace
{
// Numbers round-trip float32 precision (9 significant digits is enough for
// any IEEE-754 binary32 value) but print as a bare integer when the value is
// exactly integral — ARF's schema mixes plain integer fields (counts, ids)
// with floats, and "3" reads better than "3.000000000" for the former.
void write_number(std::string& out, double v)
{
    if (std::isfinite(v) && v == std::floor(v) &&
        std::fabs(v) < 1e15)
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", (long long)v);
        out += buf;
    }
    else
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.9g", v);
        out += buf;
    }
}

void indent_to(std::string& out, int indent, int depth)
{
    if (indent <= 0) return;
    out.push_back('\n');
    out.append((size_t)indent * depth, ' ');
}
} // namespace

void Value::write(std::string& out, int indent, int depth) const
{
    switch (kind_)
    {
        case Kind::Null:   out += "null"; break;
        case Kind::Bool:   out += bool_ ? "true" : "false"; break;
        case Kind::Number: write_number(out, num_); break;
        case Kind::String: out += escape(str_); break;

        case Kind::Array:
        {
            if (arr_.empty()) { out += "[]"; break; }
            out.push_back('[');
            for (size_t i = 0; i < arr_.size(); ++i)
            {
                indent_to(out, indent, depth + 1);
                arr_[i].write(out, indent, depth + 1);
                if (i + 1 < arr_.size()) out.push_back(',');
            }
            indent_to(out, indent, depth);
            out.push_back(']');
            break;
        }

        case Kind::Object:
        {
            if (obj_.empty()) { out += "{}"; break; }
            out.push_back('{');
            for (size_t i = 0; i < obj_.size(); ++i)
            {
                indent_to(out, indent, depth + 1);
                out += escape(obj_[i].first);
                out += (indent > 0) ? ": " : ":";
                obj_[i].second.write(out, indent, depth + 1);
                if (i + 1 < obj_.size()) out.push_back(',');
            }
            indent_to(out, indent, depth);
            out.push_back('}');
            break;
        }
    }
}

std::string Value::dump(int indent) const
{
    std::string out;
    write(out, indent, 0);
    if (indent > 0) out.push_back('\n');
    return out;
}

} // namespace arf_json
