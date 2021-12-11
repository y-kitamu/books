/*
 * factory.hpp
 * 
 * 
 * Create Date : 2020-02-25 21:16:42
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef FACTORY_HPP__
#define FACTORY_HPP__

#include <string>
#include <memory>
#include "link.hpp"
#include "tray.hpp"
#include "page.hpp"

namespace dp {

class Factory {
  public:
    virtual std::shared_ptr<Link> createLink(std::string caption, std::string url) = 0;
    virtual std::shared_ptr<Tray> createTray(std::string caption) = 0;
    virtual std::shared_ptr<Page> createPage(std::string title, std::string author) = 0;
    virtual ~Factory() {}
};

}

#endif // FACTORY_HPP__
