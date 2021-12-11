/*
 * singleton.hpp
 * 
 * 
 * Create Date : 2019-11-21 17:40:20
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef SINGLETON_HPP__
#define SINGLETON_HPP__

#include <iostream>
#include <memory>
#include <mutex>

namespace dp {

template<class T>
class Singleton {
  public:
    Singleton() = delete;
    
    template<class... Args>
    static T& getInstance(Args... args) {
        auto func = [&args...]() { create(args...); };
        std::call_once(init_flag, func);
        return *singleton;
    }

    template<class... Args>
    static void create(Args... args) {
        std::cout << "Singleton::create" << std::endl;
        singleton = std::unique_ptr<T>(new T(args...));
    }

    static void destroy() {
        delete singleton;
        singleton = nullptr;
    }

  private:
  public:
    static inline std::once_flag init_flag;
    static inline std::unique_ptr<T> singleton;
};

}

#endif // SINGLETON_HPP__
