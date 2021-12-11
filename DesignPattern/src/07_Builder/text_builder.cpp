/*
 * text_builder.cpp
 *
 * Create Date : 2020-02-24 18:04:30
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include "text_builder.hpp"

namespace dp {

void TextBuilder::makeTitle(std::string title) {
    ss << "========================================\n";
    ss << "[" << title << "]\n\n";
}

void TextBuilder::makeString(std::string str) {
    ss << "- " << str << "\n\n";
}

void TextBuilder::makeItems(std::vector<std::string> items) {
    for (auto && item : items) {
        ss << "    - " << item << "\n";
    }
    ss << "\n";
}

void TextBuilder::close() {
    ss << "========================================\n";
}

std::string TextBuilder::getResult() {
    return ss.str();
}

} // namespace dp
