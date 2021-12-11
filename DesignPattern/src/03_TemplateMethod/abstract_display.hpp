/*
 * abstract_display.hpp
 * 
 * 
 * Create Date : 2019-11-18 20:48:44
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef ABSTRACT_DISPLAY_HPP__
#define ABSTRACT_DISPLAY_HPP__

namespace dp {

class AbstractDisplay {
  public:
    virtual void open() = 0;
    virtual void print() = 0;
    virtual void close() = 0;
    void display() {
        open();
        for (int i = 0; i < 5; i++) {
            print();
        }
        close();
    }
    virtual ~AbstractDisplay() = default;
};

} // namespace dp

#endif // ABSTRACT_DISPLAY_HPP__
