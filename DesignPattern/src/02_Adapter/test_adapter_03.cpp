/*
 * test_adapter_03.cpp
 *
 * Create Date : 2019-11-17 12:26:14
 * Copyright (c) 2019 %author% <%email%>
 */

#include <memory>
#include "file_property.hpp"

int main() {
    std::unique_ptr<dp::FileIO> f(new dp::FileProperty());

    std::string src_fname = std::string(__FILE__);
    std::string abs_dir = src_fname.substr(0, src_fname.find_last_of('/'));
    std::cout << "abs dir " << abs_dir << std::endl;
    f->readFromFile(abs_dir + "/data/file.txt");
    f->setValue("year", "2014");
    f->setValue("month", "4");
    f->setValue("day", "21");
    f->writeToFile(abs_dir + "/data/newfile.txt");
}
