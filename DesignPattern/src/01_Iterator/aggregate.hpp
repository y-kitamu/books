/*
 */

#ifndef AGGREGATE_HPP__
#define AGGREGATE_HPP__

#include <vector>
#include "iterator.hpp"

namespace dp {

template<class T>
class Aggregate {
    /*
     * Iterator を生成するクラス (T の集合体)。
     * virtual 関数で継承するのが一般的かもしれないけど、
     * virtual 関数とテンプレートを同時に使用できないので、virtual 関数を使おうとすると
     * T の型を静的に与える必要が出てくるので、汎用性がなくなる。
     * なので、テンプレートを使った実装にする。
     * iterator に関係のない新しい関数を追加したい場合にはこのクラスを継承する。
     */
  public:
    Aggregate() {}
    Aggregate(int maxSize) { members = std::vector<T>(maxSize); }
    ~Aggregate() {}
    
    T getAt(int index) { return members[index]; }
    void append(T item) {
        if (members.size() == last) {
            members.emplace_back(item);
        } else {
            members[last] = item;
        }
        last++;
    }
    int getLength() { return last; }
    
    Iterator<T> iterator() { return Iterator<T>(this); };

  private:
    int last = 0;
    std::vector<T> members;
};

}

#endif // AGGREGATE_HPP__
