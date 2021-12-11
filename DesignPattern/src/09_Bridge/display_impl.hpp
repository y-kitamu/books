/*
 * display_impl.hpp
 * 
 * 
 * Create Date : 2020-03-01 00:56:15
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef DISPLAY_IMPL_HPP__
#define DISPLAY_IMPL_HPP__

namespace dp {

class DisplayImpl {
  public:
    virtual void rawOpen() = 0;
    virtual void rawPrint() = 0;
    virtual void rawClose() = 0;
    virtual ~DisplayImpl() {}
};

}

#endif // DISPLAY_IMPL_HPP__
