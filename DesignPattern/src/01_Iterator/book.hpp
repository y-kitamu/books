/*
 * book.hpp
 * 
 * 
 * Create Date : 2019-11-17 00:39:44
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef BOOK_HPP__
#define BOOK_HPP__

#include <string>

namespace dp {

class Book {
  public:
    Book() {}
    Book(std::string name) : name(name) {}
    std::string getName() { return name; }
    
  private:
    std::string name;
};

}

#endif // BOOK_HPP__
