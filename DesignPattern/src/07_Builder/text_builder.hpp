/*
 * text_builder.hpp
 * 
 * 
 * Create Date : 2020-02-24 17:58:39
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef TEXT_BUILDER_HPP__
#define TEXT_BUILDER_HPP__

#include <string>
#include <vector>
#include <sstream> 
#include "builder.hpp"

namespace dp {

class TextBuilder : public Builder {
  private:
    std::stringstream ss;
    
  public:
    void makeTitle(std::string title) override;
    void makeString(std::string str) override;
    void makeItems(std::vector<std::string> items) override;
    void close() override;
    std::string getResult() override;
    ~TextBuilder() override {};
};

} // namespace dp


#endif // TEXT_BUILDER_HPP__
