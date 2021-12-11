/*
 * test_strategy.cpp
 *
 * Create Date : 2020-03-06 23:00:13
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <iostream>
#include <memory>
#include "hand.hpp"
#include "winning_strategy.hpp"
#include "prob_strategy.hpp"
#include "player.hpp"

int main (int argc, char ** argv) {
    if (argc != 3) {
        std::cout << "Usage : ./<path>/<to>/strategy_test_strategy randomseed1 randomseed2" << std::endl;
        std::cout << "Example : /<path>/<to>/strategy_test_strategy 314 15" << std::endl;
        std::exit(0);
    }
    int seed1 = std::stoi(std::string(argv[1]));
    int seed2 = std::stoi(std::string(argv[2]));

    std::shared_ptr<dp::Player> player1 = std::make_shared<dp::Player>(
        "Taro", std::make_shared<dp::WinningStrategy>(seed1));
    std::shared_ptr<dp::Player> player2 = std::make_shared<dp::Player>(
        "Hana", std::make_shared<dp::ProbStrategy>(seed2));

    for (int i = 0; i < 10000; i++) {
        dp::Hand nextHand1 = player1->nextHand();
        dp::Hand nextHand2 = player2->nextHand();

        if (nextHand1.isStrongerThan(nextHand2)) {
            std::cout << "Winner : " + player1->name << std::endl;;
            player1->win();
            player2->loss();
        } else if (nextHand2.isStrongerThan(nextHand1)) {
            std::cout << "Winner : " + player2->name << std::endl;
            player1->loss();
            player2->win();
        } else {
            std::cout << "Even ..." << std::endl;
            player1->even();
            player2->even();
        }
    }
    std::cout << "Total result: " << std::endl;
    std::cout << player1->toString() << std::endl;
    std::cout << player2->toString() << std::endl;
}
