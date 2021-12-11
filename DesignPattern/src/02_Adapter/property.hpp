/*
 * property.hpp
 * 
 * 
 * Create Date : 2019-11-17 11:27:43
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PROPERTY_HPP__
#define PROPERTY_HPP__

#include <iostream>
#include <string>
#include <sstream>
#include <map>

namespace dp {

class Property : public std::map<std::string, std::string> {
  public:
    void load(std::stringstream &ss) {
        char delim = '=';
        std::string str;
        while (std::getline(ss, str)) {
            int split = str.find_first_of(delim);
            std::string key =  str.substr(0, split);
            // int val = std::stoi(str.substr(split + 1, str.length() - split));
            std::string val = str.substr(split + 1, str.length() - split - 1);
            std::cout << "key : " << key << ", val : " << val << std::endl;
            this->insert(std::make_pair(key, val));
        }
    }

    void store(std::stringstream &ss) {
        for (auto && pair : *this) {
            ss << pair.first << "=" << pair.second << std::endl;
        }
    }
};

}

#endif // PROPERTY_HPP__
