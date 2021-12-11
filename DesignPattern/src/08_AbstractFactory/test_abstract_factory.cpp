/*
 * test_abstract_factory.cpp
 *
 * Create Date : 2020-02-25 21:16:09
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <glog/logging.h>

#include "factory/factory.hpp"
#include "factory/link.hpp"
#include "factory/tray.hpp"
#include "factory/page.hpp"
#include "listfactory/listfactory.hpp"

int main(int argc, char ** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    
    if (argc != 2) {
        std::cout << "Usage ./<path>/<to>/abstract_factory_test_abstract_factory <class name of concrete>";
        return 0;
    }

    std::shared_ptr<dp::Factory> factory;

    if (std::string(argv[1]) == "listfactory") {
        factory = std::make_shared<dp::ListFactory>();
    }

    std::shared_ptr<dp::Link> asahi = factory->createLink("Asahi", "http://www.asahi.com");
    std::shared_ptr<dp::Link> yomiuri = factory->createLink("Yomiuri", "http://www.yomiuri.com");
    
    std::shared_ptr<dp::Link> yahoo = factory->createLink("Yahoo", "http://www.yahoo.com");
    std::shared_ptr<dp::Link> jp_yahoo = factory->createLink("Yahoo Japan", "http://www.yahoo.co.jp/");
    std::shared_ptr<dp::Link> excite = factory->createLink("Excite", "http://www.excite.com/");
    std::shared_ptr<dp::Link> google = factory->createLink("Google", "http://www.google.com/");

    std::shared_ptr<dp::Tray> traynews = factory->createTray("News Paper");
    traynews->add(asahi);
    traynews->add(yomiuri);

    std::shared_ptr<dp::Tray> trayyahoo = factory->createTray("Yahoo!");
    trayyahoo->add(yahoo);
    trayyahoo->add(jp_yahoo);

    std::shared_ptr<dp::Tray> traysearch = factory->createTray("Search engine");
    traysearch->add(trayyahoo);
    traysearch->add(excite);
    traysearch->add(google);

    std::shared_ptr<dp::Page> page = factory->createPage("LinkPage", "Y.K.");
    page->add(traynews);
    page->add(traysearch);
    page->output();
}

