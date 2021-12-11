/*
 * string_display_impl.hpp
 * 
 * 
 * Create Date : 2020-03-01 00:59:34
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef STRING_DISPLAY_IMPL_HPP__
#define STRING_DISPLAY_IMPL_HPP__

#include <iostream>
#include <string>
#include "display_impl.hpp"

namespace dp {

class StringDisplayImpl : public DisplayImpl {
  private:
    std::string str;
    int width;
  public:
    StringDisplayImpl(std::string str): str(str) {
        width = sizeof(str);
    }
    void rawOpen() override {
        printLine();
    }

    void rawPrint() override {
        std::cout << "|" << str << "|" << std::endl;
    }

    void rawClose() override {
        printLine();
    }

    void printLine() {
        std::cout << "+";
        for (int i = 0; i < width; i++) {
            std::cout << "-";
        }
        std::cout << "+" << std::endl;
    }
    ~StringDisplayImpl() {}
};

}

#endif // STRING_DISPLAY_IMPL_HPP__
