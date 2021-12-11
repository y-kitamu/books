/*
 * listpage.hpp
 * 
 * 
 * Create Date : 2020-02-29 11:16:42
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef LISTPAGE_HPP__
#define LISTPAGE_HPP__

#include "../factory/page.hpp"
#include <string>
#include <sstream>

namespace dp {

class ListPage : public Page {
  public:
    ListPage(std::string title, std::string author) : Page(title, author) {}
    std::string makeHTML() override {
        std::stringstream ss;
        ss << "<html><head><title>" << title << "</title></head>\n";
        ss << "<body>\n" << "<h1>" << title << "</h1>\n" << "<ul>\n";
        for (auto && item : content) {
            ss << item->makeHTML();
        }
        ss << "</ul>\n" << "<hr><address>" << author << "</adress>";
        ss << "</body></html>\n";
        return ss.str();
    }
};

} // namespace dp

#endif // LISTPAGE_HPP__
