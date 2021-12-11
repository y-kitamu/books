/*
 * item.hpp
 * 
 * 
 * Create Date : 2020-02-25 21:17:42
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef ITEM_HPP__
#define ITEM_HPP__

#include <string>

namespace dp {

class Item {
  public:
    std::string caption;
    Item(std::string caption) : caption(caption) {}

    virtual std::string makeHTML() = 0;
    virtual ~Item() {}
};

} // namespace dp

#endif // ITEM_HPP__
