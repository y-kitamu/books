/*
 * idcard.hpp
 * 
 * 
 * Create Date : 2019-11-20 22:11:42
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef IDCARD_HPP__
#define IDCARD_HPP__

#include <string>
#include <iostream>

#include "product.hpp"

namespace dp {

class IDCard : public Product {
  public:
    IDCard(std::string owner) : owner {owner} {
        std::cout << "make " << owner << "'s card" << std::endl;
    }
    void use() override {
        std::cout << "use " << owner << "'s card" << std::endl;
    }
    std::string getOwner() { return owner; }
    
  private:
    std::string owner;
};

}

#endif // IDCARD_HPP__
