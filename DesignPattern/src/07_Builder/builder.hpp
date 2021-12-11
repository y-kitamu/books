/*
 * builder.hpp
 * 
 * 
 * Create Date : 2020-02-24 17:37:40
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef BUILDER_HPP__
#define BUILDER_HPP__

#include <string>
#include <vector>

namespace dp {

class Builder {
  public:
    virtual void makeTitle(std::string title) = 0;
    virtual void makeString(std::string str) = 0;
    virtual void makeItems(std::vector<std::string> items) = 0;
    virtual void close() = 0;
    virtual std::string getResult() = 0;
    virtual ~Builder() {};
};

} // namespace dp

#endif // BUILDER_HPP__
