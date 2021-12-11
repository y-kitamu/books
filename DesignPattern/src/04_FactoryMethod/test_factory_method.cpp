/*
 * test_factory_method.cpp
 *
 * Create Date : 2019-11-20 22:30:58
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include <memory>
#include "idcard_factory.hpp"

int main() {
    std::unique_ptr<dp::Factory> factory(new dp::IDCardFactory());
    std::unique_ptr<dp::Product> card1 = factory->create("John");
    std::unique_ptr<dp::Product> card2 = factory->create("Tomas");
    std::unique_ptr<dp::Product> card3 = factory->create("Michael");

    card1->use();
    card2->use();
    card3->use();
}
