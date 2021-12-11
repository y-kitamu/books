/*
 * tickmaker.hpp
 * 
 * 
 * Create Date : 2019-11-21 21:13:39
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef TICKMAKER_HPP__
#define TICKMAKER_HPP__

#include <mutex>
#include <memory>

namespace dp {

class TickMaker {
  public:
    int getNextTickNumber() {
        return tick++;
    }

  private:
    int tick = 1000;
};

class TickSingleton {
  public:
    TickSingleton() = delete;
    
    static TickMaker& getInstance() {
        std::call_once(init_flag, create);
        return *tick;
    }

    static void create() {
        tick = std::unique_ptr<TickMaker>(new TickMaker());
    }

  private:
    static inline std::once_flag init_flag;
    static inline std::unique_ptr<TickMaker> tick;
};

}

#endif // TICKMAKER_HPP__
