/*
 * factory.hpp
 * 
 * 
 * Create Date : 2019-11-19 21:49:14
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef FACTORY_HPP__
#define FACTORY_HPP__

#include <string>
#include <memory>

#include "product.hpp"

namespace dp {

class Factory {
  public:
    virtual ~Factory() = default;
    
    std::unique_ptr<Product> create(std::string owner) {
        std::unique_ptr<Product> p = createProduct(owner);
        registerProduct(p);
        return p;
    }
    
  protected:
    virtual std::unique_ptr<Product> createProduct(std::string owner) = 0;
    virtual void registerProduct(std::unique_ptr<Product>& p) = 0;
};

}

#endif // FACTORY_HPP__
