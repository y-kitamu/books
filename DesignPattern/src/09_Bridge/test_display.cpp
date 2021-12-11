/*
 * test_display.hpp
 * 
 * 
 * Create Date : 2020-03-01 01:05:41
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef TEST_DISPLAY_HPP__
#define TEST_DISPLAY_HPP__

#include <memory>
#include <glog/logging.h>

#include "display.hpp"
#include "count_display.hpp"
#include "display_impl.hpp"
#include "string_display_impl.hpp"

int main(int argc, char ** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    
    std::shared_ptr<dp::Display> d1 = std::make_shared<dp::Display>(
        std::make_shared<dp::StringDisplayImpl>("Hello, Japan"));
    std::shared_ptr<dp::Display> d2 = std::make_shared<dp::CountDisplay>(
        std::make_shared<dp::StringDisplayImpl>("Hello, World"));
    std::shared_ptr<dp::CountDisplay> d3 = std::make_shared<dp::CountDisplay>(
        std::make_shared<dp::StringDisplayImpl>("Hello, Universe"));

    d1->display();
    d2->display();
    d3->multiDisplay(5);
}

#endif // TEST_DISPLAY_HPP__
