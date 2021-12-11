/*
 * product.hpp
 * 
 * 
 * Create Date : 2019-11-19 21:50:21
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PRODUCT_HPP__
#define PRODUCT_HPP__

namespace dp {

class Product {
  public:
    virtual void use() = 0;
    virtual ~Product() = default;
};

}

#endif // PRODUCT_HPP__
