/*
 * test_adapter_02.cpp
 *
 * Create Date : 2019-11-17 11:54:07
 * Copyright (c) 2019 %author% <%email%>
 */

#include "property.hpp"


int main() {
    dp::Property p;
    std::stringstream ss;
    ss << "year=2012" << std::endl
       << "month=10" << std::endl
       << "day=11" << std::endl;
    p.load(ss);

    std::cout << "Property class (key : val) : " << std::endl;
    for (auto && pair : p) {
        std::cout << pair.first << " : " << pair.second << std::endl;
    }
}
