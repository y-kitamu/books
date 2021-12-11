/*
 * idcard_factory.hpp
 * 
 * 
 * Create Date : 2019-11-20 22:17:23
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef IDCARD_FACTORY_HPP__
#define IDCARD_FACTORY_HPP__

#include <memory>
#include <vector>

#include "factory.hpp"
#include "idcard.hpp"

namespace dp {

class IDCardFactory : public Factory {
  protected:
    std::unique_ptr<Product> createProduct(std::string owner) override {
        return std::unique_ptr<Product>(new IDCard(owner));
    }
    void registerProduct(std::unique_ptr<Product>& product) override {
        owners.emplace_back(dynamic_cast<IDCard*>(product.get())->getOwner());
    }
    
  private:
    std::vector<std::string> owners;
};

}


#endif // IDCARD_FACTORY_HPP__
