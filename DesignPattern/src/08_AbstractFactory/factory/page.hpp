/*
 * page.hpp
 * 
 * 
 * Create Date : 2020-02-25 21:31:45
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PAGE_HPP__
#define PAGE_HPP__

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "item.hpp"

namespace dp {

class Page {
  protected:
    std::string title;
    std::string author;
    std::vector<std::shared_ptr<Item>> content;
  public:
    Page(std::string title, std::string author) : title(title), author(author) {}
    void add(std::shared_ptr<Item> item) {
        content.emplace_back(item);
    }
    void output() {
        std::string filename = title + ".html";
        std::ofstream ofs(filename, std::ios::out);
        ofs << makeHTML();
        ofs.close();
    }
    virtual std::string makeHTML() = 0;
    virtual ~Page() {}
};

} // namespace dp

#endif // PAGE_HPP__
