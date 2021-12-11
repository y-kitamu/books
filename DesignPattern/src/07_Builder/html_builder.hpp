/*
 * html_builder.hpp
 * 
 * 
 * Create Date : 2020-02-24 18:16:11
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef HTML_BUILDER_HPP__
#define HTML_BUILDER_HPP__

#include <string>
#include <fstream>

#include "builder.hpp"

namespace dp {

class HTMLBuilder : public Builder {
  private:
    std::string filename;
    std::ofstream ofs;
    
  public:
    void makeTitle(std::string title) override;
    void makeString(std::string str) override;
    void makeItems(std::vector<std::string> items) override;
    void close() override;
    std::string getResult() override;
    ~HTMLBuilder() override {};
};

} // namespace dp

#endif // HTML_BUILDER_HPP__
