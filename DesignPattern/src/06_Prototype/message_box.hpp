/*
 * message_box.hpp
 * 
 * 
 * Create Date : 2019-11-22 18:23:58
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef MESSAGE_BOX_HPP__
#define MESSAGE_BOX_HPP__

#include <iostream>
#include <memory>
#include "product.hpp"

namespace dp {

class MessageBox : public Product {
  public:
    MessageBox(char decochar) : decochar {decochar} {}

    void use(std::string s) override {
        int length = s.length();
        for (int i = 0; i < length + 4; i++) {
            std::cout << decochar;
        }
        std::cout << std::endl;
        
        std::cout << decochar << " " << s << " " << decochar << std::endl;

        for (int i = 0; i < length + 4; i++) {
            std::cout << decochar;
        }
        std::cout << std::endl;
    }

    std::shared_ptr<Product> createClone() override {
        std::shared_ptr<Product> p = std::make_shared<MessageBox>(*this);
        return p;
    }
    
  private:
    char decochar;
};

}

#endif // MESSAGE_BOX_HPP__
