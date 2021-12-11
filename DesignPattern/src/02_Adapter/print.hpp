/*
 * print.hpp
 * 
 * 
 * Create Date : 2019-11-17 10:25:47
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef PRINT_HPP__
#define PRINT_HPP__

namespace dp {

class Print {
    /*
     * virtual を使うと (vtbl の影響で) 関数呼び出しの overhead が大きくなるので注意が必要。
     * TODO : template を使用した Adapter の実装
     */
  public:
    virtual ~Print() = default;
    virtual void printWeak() = 0;
    virtual void printStrong() = 0;
};

}

#endif // PRINT_HPP__
