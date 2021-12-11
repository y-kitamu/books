/*
 * listlink.hpp
 * 
 * 
 * Create Date : 2020-02-27 23:31:35
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef LISTLINK_HPP__
#define LISTLINK_HPP__

#include "../factory/link.hpp"

namespace dp {

class ListLink : public Link {
  public:
    ListLink(std::string caption, std::string url) : Link(caption, url) {}
    std::string makeHTML() override {
        return "  <li><a href=\"" + url + "\">" + caption + "</a></li>\n";
    }
    ~ListLink() override {}
};

}

#endif // LISTLINK_HPP__
