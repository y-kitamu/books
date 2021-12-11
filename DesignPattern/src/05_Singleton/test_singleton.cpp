/*
 * test_singleton.cpp
 *
 * Create Date : 2019-11-21 18:37:59
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include "singleton.hpp"

class Test {
  public:
    int a = 0;
    int b = 0;
};

int main() {
    Test& t0 = dp::Singleton<Test>::getInstance<>();
    Test& t1 = dp::Singleton<Test>::getInstance<>();
    Test& t2 = dp::Singleton<Test>::getInstance<>();
    t1.a = 10;
    std::cout << t1.a << " , " << t2.a << " , " << dp::Singleton<Test>::singleton->a <<  std::endl;
}
