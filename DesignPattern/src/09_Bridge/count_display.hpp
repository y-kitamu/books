/*
 * count_display.hpp
 * 
 * 
 * Create Date : 2020-03-01 00:54:38
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef COUNT_DISPLAY_HPP__
#define COUNT_DISPLAY_HPP__

#include "display.hpp"

namespace dp {

class CountDisplay : public Display {
  public:
    using Display::Display;
    void multiDisplay(int times) {
        open();
        for (int i = 0; i < times; i++) {
            print();
        }
        close();
    }
    ~CountDisplay() {}
};

}

#endif // COUNT_DISPLAY_HPP__
