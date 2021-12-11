/*
 * test_template0.cpp
 *
 * Create Date : 2019-11-18 21:02:13
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include <memory>
#include "string_display.hpp"
#include "char_display.hpp"

int main() {
    std::unique_ptr<dp::AbstractDisplay> d1(new dp::CharDisplay('H'));
    std::unique_ptr<dp::AbstractDisplay> d2(new dp::StringDisplay("Hello world"));
    std::unique_ptr<dp::AbstractDisplay> d3(new dp::StringDisplay("Good bye"));

    d1->display();
    d2->display();
    d3->display();
}
