#ifndef SYMBOL_TABLE_HPP
#define SYMBOL_TABLE_HPP

#include <map>
#include <string>
#include <vector>

namespace la {

enum class SymbolAttribute {
    STATIC,
    FIELD,
    ARGUMENT,
    VAR,
    NONE,
};


struct Symbol {
    std::string name, type;
};


class SymbolTable {
  public:
    SymbolTable() {}

    void startSubroutine() {
        symbol_table[SymbolAttribute::ARGUMENT] = std::vector<Symbol>();
        symbol_table[SymbolAttribute::VAR] = std::vector<Symbol>();
    }

    void define(std::string name, std::string type, std::string kind) {
        symbol_table[map[kind]].emplace_back(Symbol({name, type}));
    }

    int varCount(SymbolAttribute kind) { return symbol_table[kind].size(); }

    SymbolAttribute kindOf(std::string name) {
        auto search = [this, &name](SymbolAttribute kind) {
            for (auto& symbol : this->symbol_table[kind]) {
                if (symbol.name == name) {
                    return true;
                }
            }
            return false;
        };
        for (auto& attr : attrs) {
            if (search(attr)) {
                return attr;
            }
        }
        return SymbolAttribute::NONE;
    }

    int search_index(std::string name, SymbolAttribute kind) {
        for (int i = 0; i < symbol_table[kind].size(); i++) {
            if (symbol_table[kind][i].name == name) {
                return i;
            }
        }
        return -1;
    }

    std::string typeOf(std::string name) {
        for (auto& attr : attrs) {
            auto idx = search_index(name, attr);
            if (idx >= 0) {
                return symbol_table[attr][idx].type;
            }
        }
        return "";
    }

    int indexOf(std::string name) {
        for (auto& attr : attrs) {
            auto idx = search_index(name, attr);
            if (idx >= 0) {
                return idx;
            }
        }
        return -1;
    }

  private:
    std::map<SymbolAttribute, std::vector<Symbol>> symbol_table;
    std::vector<SymbolAttribute> attrs{SymbolAttribute::ARGUMENT, SymbolAttribute::VAR,
                                       SymbolAttribute::FIELD, SymbolAttribute::STATIC};
    std::map<std::string, SymbolAttribute> map{{"argument", SymbolAttribute::ARGUMENT},
                                               {"var", SymbolAttribute::VAR},
                                               {"field", SymbolAttribute::FIELD},
                                               {"static", SymbolAttribute::STATIC},
                                               {"none", SymbolAttribute::NONE}};
};


}  // namespace la

#endif
