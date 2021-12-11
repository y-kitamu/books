#ifndef KEYWORD_HPP
#define KEYWORD_HPP

#include <map>
#include <string>

namespace la {

enum class KeyWord {
    CLASS,
    METHOD,
    FUNCTION,
    CONSTRUCTOR,
    INT,
    BOOLEAN,
    CHAR,
    VOID,
    VAR,
    STATIC,
    FIELD,
    LET,
    DO,
    IF,
    ELSE,
    WHILE,
    RETURN,
    TRUE,
    FALSE,
    NUL,
    THIS,
    NONE,
};

std::map<std::string, KeyWord> KeyWordDict{
    {"class", KeyWord::CLASS},
    {"method", KeyWord::METHOD},
    {"function", KeyWord::FUNCTION},
    {"constructor", KeyWord::CONSTRUCTOR},
    {"int", KeyWord::INT},
    {"boolean", KeyWord::BOOLEAN},
    {"char", KeyWord::CHAR},
    {"void", KeyWord::VOID},
    {"var", KeyWord::VAR},
    {"static", KeyWord::STATIC},
    {"field", KeyWord::FIELD},
    {"let", KeyWord::LET},
    {"do", KeyWord::DO},
    {"if", KeyWord::IF},
    {"else", KeyWord::ELSE},
    {"while", KeyWord::WHILE},
    {"return", KeyWord::RETURN},
    {"true", KeyWord::TRUE},
    {"false", KeyWord::FALSE},
    {"null", KeyWord::NUL},
    {"this", KeyWord::THIS},
};

}  // namespace la

#endif
