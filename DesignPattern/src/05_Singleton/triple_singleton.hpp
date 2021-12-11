/*
 * triple_singleton.hpp
 * 
 * 
 * Create Date : 2019-11-21 21:25:07
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef TRIPLE_SINGLETON_HPP__
#define TRIPLE_SINGLETON_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace dp {

class Triple {
  public:
    int test;
};

class TripleSingleton {
  public:
    TripleSingleton() = delete;

    static Triple& getInstance(int idx) {
        if (idx < 0 || 2 < idx) {
            std::cout << "out of range" << std::endl;
            throw std::out_of_range("idx " + std::to_string(idx) + " is out of range");
        }
        std::call_once(flag_init, create);
        return *triples[idx];
    }

    static void create() {
        // triples = std::make_unique<Triple>();
        for (int i = 0; i < 3; i++) {
            triples.emplace_back(std::make_unique<Triple>());
        }
    }
    
  private:
    static inline std::once_flag flag_init;
    static inline std::vector<std::unique_ptr<Triple>> triples;
    // static inline std::unique_ptr<Triple> triples;
};

}

#endif // TRIPLE_SINGLETON_HPP__
