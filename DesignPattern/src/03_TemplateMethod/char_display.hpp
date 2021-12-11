/*
 * char_display.hpp
 * 
 * 
 * Create Date : 2019-11-18 20:52:31
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef CHAR_DISPLAY_HPP__
#define CHAR_DISPLAY_HPP__

#include <iostream>
#include "abstract_display.hpp"

namespace dp {

class CharDisplay : public AbstractDisplay {
  public:
    CharDisplay(char ch) : ch{ch} {}
    void open() override {
        std::cout << "<<";
    }
    void print() override {
        std::cout << ch;
    }
    void close() override {
        std::cout << ">>" << std::endl;
    }
    
  private:
    char ch;
};

} // namespace

#endif // CHAR_DISPLAY_HPP__
