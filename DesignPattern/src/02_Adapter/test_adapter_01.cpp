/*
 * test_adapter_01.cpp
 *
 * Create Date : 2019-11-17 10:36:45
 * Copyright (c) 2019 %author% <%email%>
 */

#include "print_banner.hpp"

int main() {
    /*
     * ポリモーフィズムを実現するには基底クラスのポインタを使って継承クラスの関数を呼び出す。
     * ポインタにすることで、実行時にどのクラスの関数を実行するかを vtbl を見て判断する。
     * ポインタでなく実体を作ってしまうと、コンパイル時に呼び出し関数が決まってしまうのでうまく行かない。
     */
    dp::Print *p = new dp::PrintBanner("Hello");
    p->printWeak();
    p->printStrong();
    delete p;
}
