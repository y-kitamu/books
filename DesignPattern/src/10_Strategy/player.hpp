/*
 * player.hpp
 * 
 * 
 * Create Date : 2020-03-06 22:45:34
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PLAYER_HPP__
#define PLAYER_HPP__

#include <string>
#include <memory>
#include <sstream>
#include "strategy.hpp"

namespace dp {

class Player {
  private:
    std::shared_ptr<Strategy> strategy;
    int wincount;
    int losscount;
    int gamecount;

  public:
    std::string name;
    Player(std::string name, std::shared_ptr<Strategy> strategy): name(name), strategy(strategy) {}
    Hand nextHand() {
        return strategy->nextHand();
    }
    void win() {
        strategy->study(true);
        wincount++;
        gamecount++;
    }
    void loss() {
        strategy->study(false);
        losscount++;
        gamecount++;
    }
    void even() {
        gamecount++;
    }
    
    std::string toString() {
        std::stringstream ss;
        ss << "[" << name << ":" << gamecount << " games, "
           << wincount << " win, " << losscount << " lose]";
        return ss.str();
    }
};

} // namespace dp

#endif // PLAYER_HPP__
