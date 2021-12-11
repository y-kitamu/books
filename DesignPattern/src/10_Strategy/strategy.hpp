/*
 * strategy.hpp
 * 
 * 
 * Create Date : 2020-03-06 21:45:52
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef STRATEGY_HPP__
#define STRATEGY_HPP__

#include "hand.hpp"

namespace dp {

class Strategy {
  public:
    virtual Hand nextHand() = 0;
    virtual void study(bool win) = 0;
    virtual ~Strategy() {}
};

} // namespace dp

#endif // STRATEGY_HPP__
