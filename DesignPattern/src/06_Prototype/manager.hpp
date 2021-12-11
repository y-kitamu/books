/*
 * manager.hpp
 * 
 * 
 * Create Date : 2019-11-22 18:16:43
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef MANAGER_HPP__
#define MANAGER_HPP__

#include <string>
#include <unordered_map>
#include "product.hpp"

namespace dp {

class Manager {
  public:
    void regist(std::string name, std::shared_ptr<Product> proto) {
        showcase[name] = proto;
    }

    std::shared_ptr<Product> create(std::string protoname) {
        std::shared_ptr<Product> p = showcase[protoname];
        return p->createClone();
    }
    
  private:
    std::unordered_map<std::string, std::shared_ptr<Product>> showcase;
};

}

#endif // MANAGER_HPP__
