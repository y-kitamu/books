/*
 * winning_strategy.hpp
 * 
 * 
 * Create Date : 2020-03-06 21:47:51
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef WINNING_STRATEGY_HPP__
#define WINNING_STRATEGY_HPP__

#include <random>
#include "strategy.hpp"

namespace dp {

class WinningStrategy : public Strategy {
  public:
    WinningStrategy(int seed) : rd(seed) {}
    Hand nextHand() override {
        if (!won) {
            return Hand::getHand(dist(rd));
        }
        return prev_hand;
    }

    void study(bool win) override {
        won = win;
    }

    ~WinningStrategy() {}

  private:
    std::default_random_engine rd;
    std::uniform_int_distribution<> dist {0, 2};
    bool won = false;
    Hand prev_hand = Hand::getHand(0);
};

}

#endif // WINNING_STRATEGY_HPP__
