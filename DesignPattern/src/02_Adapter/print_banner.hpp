/*
 * print_banner.hpp
 * 
 * 
 * Create Date : 2019-11-17 10:28:21
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PRINT_BANNER_HPP__
#define PRINT_BANNER_HPP__

#include "banner.hpp"
#include "print.hpp"

namespace dp {

class PrintBanner : public Banner, public Print {
  public:
    PrintBanner(std::string str) : Banner(str) {}
    void printWeak() override { showWithParen(); }
    void printStrong() override { showWithAster(); }
    //~PrintBanner() override {};
};

} // namespace dp

#endif // PRINT_BANNER_HPP__
