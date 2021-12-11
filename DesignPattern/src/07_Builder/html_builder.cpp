/*
 * html_builder.cpp
 *
 * Create Date : 2020-02-24 18:24:08
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <iostream>
#include "html_builder.hpp"

namespace dp {

void HTMLBuilder::makeTitle(std::string title) {
    filename = title + ".html";
    ofs.open(filename, std::ios::out);
    if (!ofs.is_open()) {
        std::cout << "failed to open : " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    ofs << "<html><head><title>" << title << "</title></head><body>";
    ofs << "<h1>" << title << "</h1>";
}

void HTMLBuilder::makeString(std::string str) {
    ofs << "<p>" << str << "</p>";
}

void HTMLBuilder::makeItems(std::vector<std::string> items) {
    ofs << "<ul>";
    for (auto && item : items) {
        ofs << "<li>" << item << "</li>";
    }
    ofs << "</ul>";
}

void HTMLBuilder::close() {
    ofs << "</body></html>";
    ofs.close();
}

std::string HTMLBuilder::getResult() {
    return filename;
}

} // namespace dp
