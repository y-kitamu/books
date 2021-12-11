/*
 * string_display.hpp
 * 
 * 
 * Create Date : 2019-11-18 20:55:43
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef STRING_DISPLAY_HPP__
#define STRING_DISPLAY_HPP__

#include <iostream>
#include <string>

#include "abstract_display.hpp"

namespace dp {

class StringDisplay : public AbstractDisplay {
  public:
    StringDisplay(std::string str) : str {str}, width {(int)str.length()} {}
    void open() override { printLine(); }
    void close() override { printLine(); }
    void print() override {
        std::cout << "|" << str << "|" << std::endl;
    }
    
  private:
    void printLine() {
        std::cout << "+";
        for (size_t i = 0; i < str.length(); i++) {
            std::cout << "-";
        }
        std::cout << "+" << std::endl;
    }

    std::string str;
    int width;
};

} // namespace dp

#endif // STRING_DISPLAY_HPP__
