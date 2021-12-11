#ifndef TOKEN_TYPE_HPP
#define TOKEN_TYPE_HPP

namespace la {

enum class TokenType {
    KEYWORD,
    SYMBOL,
    IDENTIFIER,
    INT_CONST,
    STRING_CONST,
    NONE,
};

}  // namespace la

#endif
