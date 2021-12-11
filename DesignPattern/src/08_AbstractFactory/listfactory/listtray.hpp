/*
 * listtray.hpp
 * 
 * 
 * Create Date : 2020-02-27 23:34:25
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef LISTTRAY_HPP__
#define LISTTRAY_HPP__

#include <sstream>
#include "../factory/tray.hpp"

namespace dp {

class ListTray : public Tray {
  public:
    ListTray(std::string caption): Tray(caption) {}
    std::string makeHTML() override {
        std::stringstream buffer;
        buffer << "<li>\n" << caption << "\n" << "<ul>\n";
        for (auto && tray : array) {
            buffer << tray->makeHTML();
        }
        buffer << "</ul>\n" << "</li>\n";
        return buffer.str();
    }

    ~ListTray() override {}
};

}

#endif // LISTTRAY_HPP__
