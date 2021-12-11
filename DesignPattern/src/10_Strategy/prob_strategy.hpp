/*
 * prob_strategy.hpp
 * 
 * 
 * Create Date : 2020-03-06 22:11:04
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PROB_STRATEGY_HPP__
#define PROB_STRATEGY_HPP__

#include <random>
#include "strategy.hpp"

namespace dp {

class ProbStrategy : public Strategy {
  private:
    std::default_random_engine rd;
    int prevHandValue = 0;
    int currentHandValue = 0;

    std::vector<std::vector<int>> history = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

  public:
    ProbStrategy(int seed) : rd(seed) {}
    Hand nextHand() override {
        std::uniform_int_distribution<> dist {0, getSum(currentHandValue)};
        int bet = dist(rd);
        int handvalue = 0;
        if (bet < history[currentHandValue][0]) {
            handvalue = 0;
        } else if (bet < history[currentHandValue][0] + history[currentHandValue][1]) {
            handvalue = 1;
        } else {
            handvalue = 2;
        }
        prevHandValue = currentHandValue;
        currentHandValue = handvalue;
        return Hand::getHand(handvalue);
    }

    void study(bool win) override {
        if (win) {
            history[prevHandValue][currentHandValue]++;
        } else {
            history[prevHandValue][(currentHandValue + 1) % 3]++;
            history[prevHandValue][(currentHandValue + 2) % 3]++;
        }
    }

    ~ProbStrategy() {}
    
  private:
    int getSum(int hv) {
        int sum = 0;
        for (int i = 0; i < 3; i++) {
            sum += history[hv][i];
        }
        return sum;
    }
};

} // namespace dp

#endif // PROB_STRATEGY_HPP__
