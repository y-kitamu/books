/*
 * test_prototype.cpp
 *
 * Create Date : 2019-11-22 18:50:04
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#include "manager.hpp"
#include "message_box.hpp"
#include "underline_pen.hpp"

int main() {
    dp::Manager manager;
    std::shared_ptr<dp::Product> upen = std::make_shared<dp::UnderlinePen>('~');
    std::shared_ptr<dp::Product> mbox = std::make_shared<dp::MessageBox>('*');
    std::shared_ptr<dp::Product> sbox = std::make_shared<dp::MessageBox>('/');
    manager.regist("strong message", upen);
    manager.regist("warning box", mbox);
    manager.regist("slash box", sbox);

    std::shared_ptr<dp::Product> p1 = manager.create("strong message");
    std::shared_ptr<dp::Product> p2 = manager.create("warning box");
    std::shared_ptr<dp::Product> p3 = manager.create("slash box");
    p1->use("Hello, world");
    p2->use("Hello, world");
    p3->use("Hello, world");
}
