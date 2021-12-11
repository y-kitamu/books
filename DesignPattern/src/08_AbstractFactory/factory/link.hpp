/*
 * link.hpp
 * 
 * 
 * Create Date : 2020-02-25 21:22:43
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef LINK_HPP__
#define LINK_HPP__

#include <string>
#include "item.hpp"

namespace dp {

class Link : public Item {
  protected:
    std::string url;
  public:
    Link(std::string caption, std::string url) : Item(caption) {
        url = url;
    }
    ~Link() override {}
};

} // namespace dp

#endif // LINK_HPP__
