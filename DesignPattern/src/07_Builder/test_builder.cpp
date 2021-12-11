/*
 * test_builder.cpp
 *
 * Create Date : 2020-02-24 21:02:17
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <string>
#include <iostream>
#include "Director.hpp"
#include "text_builder.hpp"
#include "html_builder.hpp"

void usage() {
    std::cout << "Usage : ./<path>/<to>/builder_test_builder plain  # plain text document" << std::endl;
    std::cout << "Usage : ./<path>/<to>/builder_test_builder html # html document" << std::endl;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        usage();
        return 0;
    }

    std::shared_ptr<dp::Builder> builder;
    if (std::string(argv[1]) == "plain") {
        builder = std::make_shared<dp::TextBuilder>();
    } else if (std::string(argv[1]) == "html") {
        builder = std::make_shared<dp::HTMLBuilder>();
    } else {
        usage();
        return 0;
    }
    dp::Director director(builder);
    director.construct();
    std::string result = builder->getResult();
    std::cout << result << std::endl;
}
