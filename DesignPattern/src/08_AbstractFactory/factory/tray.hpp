/*
 * tray.hpp
 * 
 * 
 * Create Date : 2020-02-25 21:27:40
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef TRAY_HPP__
#define TRAY_HPP__

#include <memory>
#include <vector>
#include "item.hpp"

namespace dp {

class Tray : public Item {
  protected:
    std::vector<std::shared_ptr<Item>> array;
  public:
    Tray(std::string caption) : Item(caption) {}
    void add(std::shared_ptr<Item> item) {
        array.emplace_back(item);
    }
    ~Tray() override {}
};

} // namespace dp

#endif // TRAY_HPP__
