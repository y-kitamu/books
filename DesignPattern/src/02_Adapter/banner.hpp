/*
 * banner.hpp
 * 
 * 
 * Create Date : 2019-11-17 10:23:12
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef BANNER_HPP__
#define BANNER_HPP__

#include <iostream>
#include <string>

namespace dp {

class Banner {
  public:
    Banner(std::string str) : str(str) {}
    void showWithParen() { std::cout << "(" << str << ")" << std::endl; }
    void showWithAster() { std::cout << "*" << str << "*" << std::endl; }
    
  private:
    std::string str;
};

}

#endif // BANNER_HPP__
