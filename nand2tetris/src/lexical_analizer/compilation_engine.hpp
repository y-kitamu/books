#ifndef COMPILATION_ENGINE_HPP
#define COMPILATION_ENGINE_HPP

#include <cassert>
#include <ostream>
#include <string>

#include "lexical_analizer/jack_tokenizer.hpp"
#include "lexical_analizer/symbol_table.hpp"
#include "lexical_analizer/token_type.hpp"

namespace la {

class CompilationEngine {
  public:
    CompilationEngine(std::string input_filename, std::string output_filename)
        : tokenizer(input_filename), ofs(output_filename) {}

    bool advance_and_assertion_check(la::TokenType&& token_type) {
        if (tokenizer.hasMoreTokens()) {
            tokenizer.advance();
            assert(token_type == tokenizer.tokenType());
            return token_type == tokenizer.tokenType();
        }
        return false;
    }

    std::string get_indent_str() {
        std::string indent_str = "";
        for (int i = 0; i < indent; i++) {
            indent_str += " ";
        }
        return indent_str;
    }

    void compileClass() {
        std::string tag = "class";
        compilePrefixTag(tag);

        // keyword : 'class'
        compile();
        // identifier : class name
        compile();
        // symbol {
        compile();
        // class body
        compileClassVarDec();
        compileSubroutine();
        // symbol }
        compile();

        compileSuffixTag(tag);
    }

    void compileClassVarDec() {
        std::string tag = "classVarDec";
        std::vector<std::string> kwd_vardec{"static", "field"};

        while (true) {
            if (advance_and_assertion_check(la::TokenType::KEYWORD) &&
                contains(tokenizer.keyWord(false), kwd_vardec)) {
                compilePrefixTag(tag);
                // keyword static or field
                compile();
                // keyword or identifier (variable type)
                compile();
                // var names
                while (true) {
                    // identifier (variable name);
                    compile();
                    // symbol (';' or ',')
                    compile();
                    if (tokenizer.symbol() == ";") {
                        break;
                    }
                }
                compileSuffixTag(tag);
            } else {
                break;
            }
        }
    }

    void compileSubroutine() {
        symbol_table.startSubroutine();
        std::string tag = "subroutineDec";
        std::string tag_body = "subroutineBody";
        std::vector<std::string> kwd_subroutine{"constructor", "function", "method"};
        std::vector<std::string> kwd_vardec{"var"};
        std::vector<std::string> kwd_state{"let", "if", "else", "while", "do", "return"};

        while (advance_and_assertion_check(TokenType::KEYWORD) &&
               contains(tokenizer.keyWord(false), kwd_subroutine)) {
            compilePrefixTag(tag);

            // keyword (function type)
            compile();
            // keyword or identifier (return type)
            compile();
            // identifier (function name)
            compile();
            // parameter list
            compileParameterList();
            // function body
            compilePrefixTag(tag_body);
            // symbol '{'
            compile();
            while (!(advance_and_assertion_check(TokenType::SYMBOL) && tokenizer.symbol(false) == "}")) {
                if (contains(tokenizer.keyWord(false), kwd_vardec)) {
                    compileVarDec();
                } else if (contains(tokenizer.keyWord(false), kwd_state)) {
                    compileStatements();
                } else {
                    break;
                }
            }
            // symbol '}'
            compile();

            compileSuffixTag(tag_body);
            compileSuffixTag(tag);
        }
    }

    void compileParameterList() {
        std::string tag = "parameterList";

        // symbol '('
        compile();
        compilePrefixTag(tag);

        while (!(advance_and_assertion_check(TokenType::SYMBOL) && tokenizer.symbol(false) == ")")) {
            if (tokenizer.tokenType() == TokenType::SYMBOL) {
                // symbol ',';
                compile();
            }
            // keyword or identifier (parameter type)
            compile();
            // identifier (parameter name)
            compile();
        }

        compileSuffixTag(tag);
        // symbol ')'
        compile();
    }

    void compileVarDec() {
        std::string tag = "varDec";
        compilePrefixTag(tag);

        // keyword 'var'
        compile();
        // type
        compile();

        while (true) {
            // variable name
            compile();
            // symbol ',' or ';'
            compile();
            if (tokenizer.symbol() == ";") {
                break;
            }
        }
        compileSuffixTag(tag);
    }

    void compileStatements() {
        std::string tag = "statements";
        compilePrefixTag(tag);

        while (true) {
            // std::cout << tokenizer.get_current_token() << std::endl;
            std::string kwd = tokenizer.keyWord(false);
            if (kwd == "let") {
                compileLet();
            } else if (kwd == "if") {
                compileIf();
            } else if (kwd == "while") {
                compileWhile();
            } else if (kwd == "do") {
                compileDo();
            } else if (kwd == "return") {
                compileReturn();
            }
            // std::cout << tokenizer.get_current_token() << std::endl;
            if (advance_and_assertion_check(TokenType::SYMBOL) && tokenizer.symbol(false) == "}") {
                break;
            }
        }
        compileSuffixTag(tag);
    }

    void compileDo() {
        std::string tag = "doStatement";
        compilePrefixTag(tag);
        // keyword do
        compile();
        while (true) {
            // identifier (class or subroutine name)
            compile();
            // symbol '.' or '('
            compile();
            if (tokenizer.symbol() == "(") {
                break;
            }
        }
        compileExpressionList();
        // symbol ')'
        compile();
        // symbol ';'
        compile();
        compileSuffixTag(tag);
    }

    void compileLet() {
        std::string tag = "letStatement";
        compilePrefixTag(tag);

        // keyword 'let'
        compile();
        // identifier (variable name)
        compile();

        tokenizer.advance();
        std::string symbol = tokenizer.symbol(false);
        while (symbol == "[") {
            // symbol [
            compile();
            compileExpression();
            // symbol (] or [)
            compile();
            tokenizer.advance();
            symbol = tokenizer.symbol(false);
        }

        // '='
        compile();
        compileExpression();
        // ';
        compile();

        compileSuffixTag(tag);
    }

    void compileWhile() {
        std::string tag = "whileStatement";
        compilePrefixTag(tag);

        // keyword while
        compile();
        // symbol '('
        compile();
        compileExpression();
        // symbol ')'
        compile();
        // symbol '{'
        compile();
        tokenizer.advance();
        compileStatements();
        // symbol '}'
        compile();

        compileSuffixTag(tag);
    }

    void compileReturn() {
        std::string tag = "returnStatement";
        compilePrefixTag(tag);
        // keyword return;
        compile();
        tokenizer.advance();
        if (tokenizer.tokenType() != TokenType::SYMBOL || tokenizer.symbol(false) != ";") {
            compileExpression();
        }
        // symbol ;
        compile();
        compileSuffixTag(tag);
    }

    void compileIf() {
        std::string tag = "ifStatement";
        compilePrefixTag(tag);

        // keyword if
        compile();
        // symbol (
        compile();
        compileExpression();
        // symbol )
        compile();
        // symbol {
        compile();
        tokenizer.advance();
        compileStatements();
        // symbol }
        compile();

        tokenizer.advance();
        if (tokenizer.tokenType() == TokenType::KEYWORD && tokenizer.keyWord(false) == "else") {
            // keyword else;
            compile();
            // symbol "{"
            compile();
            tokenizer.advance();
            compileStatements();
            // symbol '}'
            compile();
        }
        compileSuffixTag(tag);
    }

    void compileExpression() {
        std::string tag = "expression";
        std::vector<std::string> ops = {"+", "-", "*", "/", "&amp;", "|", "&lt;", "&gt;", "="};
        compilePrefixTag(tag);

        bool flag = true;
        while (flag) {
            compileTerm();
            tokenizer.advance();
            flag = tokenizer.tokenType() == TokenType::SYMBOL && contains(tokenizer.symbol(false), ops);
            if (flag) {
                // symbol : one of the ops
                compile();
            }
        }
        compileSuffixTag(tag);
    }

    void compileTerm() {
        std::string tag = "term";
        std::vector<std::string> uops = {"-", "~"};
        compilePrefixTag(tag);
        // int const, str const, keyword, identifier or symbol;
        compile();
        auto token_type = tokenizer.tokenType();
        if (token_type == TokenType::SYMBOL) {
            if (contains(tokenizer.symbol(false), uops)) {
                // symbol : uops
                compile();
                compileTerm();
            } else if (tokenizer.symbol(false) == "(") {
                // symbol '('
                compile();
                compileExpression();
                // symbol ')'
                compile();
            }
        } else if (token_type == TokenType::IDENTIFIER) {
            tokenizer.advance();
            if (tokenizer.tokenType() == TokenType::SYMBOL) {
                auto symbol = tokenizer.symbol(false);
                if (symbol == "[") {
                    // symbol "["
                    compile();
                    compileExpression();
                    // symbol "]"
                    compile();
                }
                if (symbol == "." || symbol == "(") {
                    // symbol '.' or '('
                    compile();
                    while (tokenizer.symbol() != "(") {
                        // identifier (class or subroutine name)
                        compile();
                        // symbol '.' or '('
                        compile();
                    }
                    compileExpressionList();
                    // symbol ')'
                    compile();
                }
            }
        }
        compileSuffixTag(tag);
    }

    void compileExpressionList() {
        std::string tag = "expressionList";
        compilePrefixTag(tag);

        while (true) {
            tokenizer.advance();
            if (tokenizer.tokenType() == TokenType::SYMBOL) {
                if (tokenizer.symbol(false) == ")") {
                    break;
                } else if (tokenizer.symbol(false) == ",") {
                    // symbol; ","
                    compile();
                }
            }
            compileExpression();
        }

        compileSuffixTag(tag);
    }

    void compilePrefixTag(std::string tagname) {
        std::string str = fmt::format("{}<{}>\n", get_indent_str(), tagname);
        if (debug) {
            std::cout << str;
        }
        ofs << str;
        indent += 2;
    }

    void compileSuffixTag(std::string tagname) {
        indent -= 2;
        std::string str = fmt::format("{}</{}>\n", get_indent_str(), tagname);
        if (debug) {
            std::cout << str;
        }
        ofs << str;
    }

    void compile(TokenType expect = TokenType::NONE) {
        std::string key, value;
        tokenizer.advance();
        auto token_type = tokenizer.tokenType();
        assert(expect == token_type || expect == TokenType::NONE);
        switch (token_type) {
            case TokenType::IDENTIFIER:
                key = "identifier";
                value = tokenizer.identifier();
                break;
            case TokenType::KEYWORD:
                key = "keyword";
                value = tokenizer.keyWord();
                break;
            case TokenType::SYMBOL:
                key = "symbol";
                value = tokenizer.symbol();
                break;
            case TokenType::INT_CONST:
                key = "integerConstant";
                value = std::to_string(tokenizer.intVal());
                break;
            case TokenType::STRING_CONST:
                key = "stringConstant";
                value = tokenizer.stringVal();
                break;
            default:
                return;
        }
        std::string indent_str = get_indent_str();
        std::string str = fmt::format("{0}<{1}> {2} </{1}>\n", indent_str, key, value);
        if (debug) {
            std::cout << str;
        }
        ofs << str;
    }


    void run_compile(bool debug = false) {
        this->debug = debug;
        compileClass();
    }

  private:
    JackTokenizer tokenizer;
    SymbolTable symbol_table;
    std::string id_category, id_name, id_attr;
    bool id_defined;
    int id_idx;
    std::ofstream ofs;
    int indent = 0;
    bool debug = false;
};

}  // namespace la

#endif
