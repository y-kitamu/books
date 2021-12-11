/*
 * product.hpp
 * 
 * 
 * Create Date : 2019-11-22 18:04:18
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PRODUCT_HPP__
#define PRODUCT_HPP__

#include <string>
#include <memory>

namespace dp {

class Product {
  public:
    virtual void use(std::string s) = 0;
    virtual std::shared_ptr<Product> createClone() = 0;
};

} // namespace dp

#endif // PRODUCT_HPP__
