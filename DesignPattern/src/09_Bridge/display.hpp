/*
 * display.hpp
 * 
 * 
 * Create Date : 2020-03-01 00:49:28
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef DISPLAY_HPP__
#define DISPLAY_HPP__

#include <memory>
#include "display_impl.hpp"

namespace dp {

class Display {
  private:
    std::shared_ptr<DisplayImpl> impl;
  public:
    Display(std::shared_ptr<DisplayImpl> impl): impl(impl) {}
    virtual void open() {
        impl->rawOpen();
    }
    virtual void print() {
        impl->rawPrint();
    }
    virtual void close() {
        impl->rawClose();
    }
    void display() {
        open();
        print();
        close();
    }
    virtual ~Display() {}
};

} // namespace dp

#endif // DISPLAY_HPP__
