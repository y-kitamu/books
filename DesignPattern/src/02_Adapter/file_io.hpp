/*
 * file_io.hpp
 * 
 * 
 * Create Date : 2019-11-17 11:27:19
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef FILE_IO_HPP__
#define FILE_IO_HPP__

#include <string>

namespace dp {

class FileIO {
  public:
    virtual void readFromFile(std::string filename) = 0;
    virtual void writeToFile(std::string filename) = 0;
    virtual void setValue(std::string key, std::string value) = 0;
    virtual std::string getValue(std::string key) = 0;
    virtual ~FileIO() = default;
};

}

#endif // FILE_IO_HPP__
