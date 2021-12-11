/*
 * Director.hpp
 * 
 * 
 * Create Date : 2020-02-24 17:47:06
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef DIRECTOR_HPP__
#define DIRECTOR_HPP__

#include <memory>
#include "builder.hpp"


namespace dp {

class Director {
  private:
    std::shared_ptr<Builder> builder;
  public:
    Director(std::shared_ptr<Builder> builder) : builder(builder) {}
    void construct() {
        builder->makeTitle("Greeting");
        builder->makeString("from morning till noon");
        builder->makeItems({"Good morning", "Good afternoon"});
        builder->makeString("at night");
        builder->makeItems({"Good night", "Good bye"});
        builder->close();
    }
};

} // namespace dp

#endif // DIRECTOR_HPP__
