/*
 * file_property.hpp
 * 
 * 
 * Create Date : 2019-11-17 11:27:34
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef FILE_PROPERTY_HPP__
#define FILE_PROPERTY_HPP__

#include <fstream>
#include "property.hpp"
#include "file_io.hpp"

namespace dp {

class FileProperty : public Property, public FileIO {
  public:
    void readFromFile(std::string filename) override {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            std::cout << "failed to open " << filename << std::endl;
            return;
        }
        
        std::stringstream ss;
        ss << ifs.rdbuf();
        ifs.close();

        load(ss);
    }

    void writeToFile(std::string filename) override {
        std::stringstream ss;
        store(ss);
        
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cout << "failed to open " << filename << std::endl;
            return;
        }

        ofs << ss.rdbuf();
        ofs.close();
    }

    void setValue(std::string key, std::string value) override {
        //this->insert(std::make_pair(key, value));
        (*this)[key] = value;
    }

    std::string getValue(std::string key) override {
        return (*this)[key];
    }
};

}

#endif // FILE_PROPERTY_HPP__
