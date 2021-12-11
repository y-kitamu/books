/*
 * listfactory.hpp
 * 
 * 
 * Create Date : 2020-02-27 23:25:36
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef LISTFACTORY_HPP__
#define LISTFACTORY_HPP__

#include "../factory/factory.hpp"
#include "listlink.hpp"
#include "listtray.hpp"
#include "listpage.hpp"

namespace dp {

class ListFactory : public Factory {
  public:
    std::shared_ptr<Link> createLink(std::string caption, std::string url) override {
        return std::make_shared<ListLink>(caption, url);
    }

    std::shared_ptr<Tray> createTray(std::string caption) override {
        return std::make_shared<ListTray>(caption);
    }

    std::shared_ptr<Page> createPage(std::string title, std::string author) override {
        return std::make_shared<ListPage>(title, author);
    }

    ~ListFactory() {}
};

}

#endif // LISTFACTORY_HPP__
