/*
 * test_singleton1.cpp
 *
 * Create Date : 2019-11-21 21:20:25
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include <iostream>
#include "tickmaker.hpp"

int main() {
    dp::TickMaker& t1 = dp::TickSingleton::getInstance();
    dp::TickMaker& t2 = dp::TickSingleton::getInstance();

    for (int i = 0; i < 10; i++) {
        std::cout << t1.getNextTickNumber() << ", " << t2.getNextTickNumber() << std::endl;
    }
}
