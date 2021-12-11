/*
 * test_singleton2.cpp
 *
 * Create Date : 2019-11-21 22:02:07
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include "triple_singleton.hpp"

int main() {
    dp::Triple& trip0 = dp::TripleSingleton::getInstance(0);
    dp::Triple& trip1 = dp::TripleSingleton::getInstance(1);
    dp::Triple& trip3 = dp::TripleSingleton::getInstance(3);

    // if (trip0 == trip1) {
        
    // }
}
