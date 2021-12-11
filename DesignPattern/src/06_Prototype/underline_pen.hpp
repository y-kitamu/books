/*
 * underline_pen.hpp
 * 
 * 
 * Create Date : 2019-11-22 18:44:52
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef UNDERLINE_PEN_HPP__
#define UNDERLINE_PEN_HPP__

#include <iostream>
#include <memory>
#include "product.hpp"

namespace dp {

class UnderlinePen : public Product {
  public:
    UnderlinePen(char ulchar) : ulchar {ulchar} {};
    void use(std::string s) override {
        int length = s.length();
        std::cout << "\"" + s << "\"" << std::endl;
        std::cout << " ";
        for (int i = 0; i < length; i++) {
            std::cout << ulchar;
        }
        std::cout << std::endl;
    }
    std::shared_ptr<Product> createClone() override {
        return std::make_shared<UnderlinePen>(*this);
    }

  private:
    char ulchar;
};

}

#endif // UNDERLINE_PEN_HPP__
