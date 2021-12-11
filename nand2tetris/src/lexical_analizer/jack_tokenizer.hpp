#ifndef JACK_TOKENIZER_HPP
#define JACK_TOKENIZER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "fmt/core.h"

#include "lexical_analizer/keyword.hpp"
#include "lexical_analizer/token_type.hpp"


namespace ba = boost::algorithm;
namespace fs = boost::filesystem;

namespace la {


bool contains(std::string cand, const std::vector<std::string>& tokens) {
    for (auto token : tokens) {
        if (cand == token) {
            return true;
        }
    }
    return false;
}


class JackTokenizer {
  public:
    JackTokenizer(std::string input_file) : ifs(input_file), input_file(input_file) {
        if (!ifs.is_open()) {
            fmt::print("Failed to open file {}\n", input_file);
            fmt::print("Abort.\n");
            std::exit(0);
        }
        input_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        std::vector<std::string> splits;
        boost::algorithm::split(splits, input_str, boost::is_any_of("\n"));

        // remove comment
        std::regex re_com{R"(//.*)"};
        for (auto& split : splits) {
            split = std::regex_replace(split, re_com, "");
            split = std::regex_replace(split, std::regex("[[:cntrl:]]"), "");
        }

        input_str = join(splits, " ");
        re_com = std::regex(R"(/\*.*?\*/)");
        input_str = std::regex_replace(input_str, re_com, " ");
        // std::cout << input_str << std::endl;

        bool in_str_const = false;
        bool in_token = false;
        int token_start = 0;
        int token_end = 0;
        std::vector<std::pair<int, int>> token_pos;
        for (int idx = 0; idx < input_str.size(); idx++) {
            // STRING_CONST
            if (input_str[idx] == '"') {
                in_token = false;
                if (in_str_const) {
                    token_end = std::max(idx, token_start);
                    token_pos.emplace_back(std::make_pair(token_start, token_end + 1));
                } else {
                    token_start = idx;
                }
                in_str_const = !in_str_const;
                continue;
            }
            if (in_str_const) {
                continue;
            }

            // white space
            if (input_str[idx] == ' ') {
                if (in_token && token_start <= token_end) {
                    token_pos.emplace_back(token_start, token_end + 1);
                }
                in_token = false;
                token_start = idx + 1;
                continue;
            }

            if (contains(std::string({input_str[idx]}), SYMBOLS)) {
                if (in_token) {
                    // prev token
                    if (token_start <= token_end) {
                        token_pos.emplace_back(token_start, token_end + 1);
                    }
                }
                // symbols
                token_pos.emplace_back(idx, idx + 1);

                // next token;
                token_start = idx + 1;
                in_token = false;
            } else {
                // normal characters
                token_end = idx;
                in_token = true;
            }
        }

        if (token_start < token_end) {
            token_pos.emplace_back(token_start, token_end);
        }
        tokens = std::vector<std::string>(token_pos.size());
        for (int i = 0; i < token_pos.size(); i++) {
            auto pos = token_pos[i];
            tokens[i] = input_str.substr(pos.first, pos.second - pos.first);
        }
    }

    std::string join(std::vector<std::string> strings, std::string deli) {
        std::stringstream ss;
        for (int i = 0; i < strings.size(); i++) {
            ss << strings[i];
            if (i < strings.size() - 1 && strings[i] != "") {
                ss << deli;
            }
        }
        return ss.str();
    }

    void run_tokenize(std::string output_fname) {
        auto dirpath = fs::path(output_fname).parent_path();
        if (!fs::exists(dirpath)) {
            fs::create_directories(dirpath);
        }
        // std::stringstream ss;
        std::ofstream ofs(output_fname);
        if (!ofs.is_open()) {
            std::cout << "Failed to open output file : " << output_fname << std::endl;
        }
        ofs << "<tokens>" << std::endl;
        while (hasMoreTokens()) {
            advance();
            // std::cout << current_token << std::endl;
            switch (tokenType()) {
                case TokenType::KEYWORD:
                    ofs << "<keyword> " << keyWord() << " </keyword>" << std::endl;
                    break;
                case TokenType::SYMBOL:
                    ofs << "<symbol> " << symbol() << " </symbol>" << std::endl;
                    break;
                case TokenType::IDENTIFIER:
                    ofs << "<identifier> " << identifier() << " </identifier>" << std::endl;
                    break;
                case TokenType::INT_CONST:
                    ofs << "<integerConstant> " << intVal() << " </integerConstant>" << std::endl;
                    break;
                case TokenType::STRING_CONST:
                    ofs << "<stringConstant> " << stringVal() << " </stringConstant>" << std::endl;
                    break;
                case TokenType::NONE:
                    std::cout << "Token type is NONE : " << current_token << std::endl;
                    return;
            }
        }
        ofs << "</tokens>" << std::endl;
    }

    bool hasMoreTokens() { return !is_token_used || cur_idx < tokens.size(); }

    std::string advance() {
        if (!is_token_used) {
            return current_token;
        }
        current_token = tokens[cur_idx++];
        is_token_used = false;
        return current_token;
    }

    TokenType tokenType() {
        if (contains(current_token, KEYWORDS)) {
            current_token_type = TokenType::KEYWORD;
        } else if (contains(current_token, SYMBOLS)) {
            current_token_type = TokenType::SYMBOL;
        } else if (current_token[0] == '"' && current_token[current_token.size() - 1] == '"') {
            current_token_type = TokenType::STRING_CONST;
        } else if (std::all_of(current_token.begin(), current_token.end(), isdigit)) {
            current_token_type = TokenType::INT_CONST;
        } else {
            current_token_type = TokenType::IDENTIFIER;
        }
        return current_token_type;
    }

    std::string keyWord(bool is_used = true) {
        if (tokenType() != TokenType::KEYWORD) {
            throw std::invalid_argument("Token type must be TokenType::KEYWORD.");
        }
        is_token_used = true && is_used;
        return current_token;
    }

    std::string symbol(bool is_used = true) {
        if (tokenType() != TokenType::SYMBOL) {
            throw std::invalid_argument("Token type must be TokenType::SYMBOL.");
        }

        is_token_used = true && is_used;
        if (current_token == "<") {
            return "&lt;";
        } else if (current_token == ">") {
            return "&gt;";
        } else if (current_token == "&") {
            return "&amp;";
        }
        return current_token;
    }

    std::string identifier(bool is_used = true) {
        if (tokenType() != TokenType::IDENTIFIER) {
            throw std::invalid_argument("Token type must be TokenType::IDENTIFIER");
        }
        is_token_used = true && is_used;
        return current_token;
    }

    int intVal(bool is_used = true) {
        if (tokenType() != TokenType::INT_CONST) {
            throw std::invalid_argument("Token type must be TokenType::INT_CONST");
        }
        is_token_used = true && is_used;
        return std::stoi(current_token);
    }

    std::string stringVal(bool is_used = true) {
        if (tokenType() != TokenType::STRING_CONST) {
            throw std::invalid_argument("Token type must be TokenType::STRING_CONST");
        }
        is_token_used = true && is_used;
        return current_token.substr(1, current_token.length() - 2);
    }

    std::string get_current_token() { return current_token; }

  private:
    const std::vector<std::string> SEPARATORS{" ", "\n"};
    const std::vector<std::string> KEYWORDS{
        "class", "constructor", "function", "method", "field", "static", "var",
        "int",   "char",        "boolean",  "void",   "true",  "false",  "null",
        "this",  "let",         "do",       "if",     "else",  "while",  "return"};
    const std::vector<std::string> SYMBOLS{"{", "}", "(", ")", "[", "]", ".", ",", ";", "+",
                                           "-", "*", "/", "&", "|", "<", ">", "=", "~"};
    std::ifstream ifs;
    std::string input_file;
    std::string input_str, current_token, next_token;
    std::vector<std::string> tokens;
    bool is_token_used = true;
    int cur_idx = 0;
    TokenType current_token_type;
};

}  // namespace la

#endif
