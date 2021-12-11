/*
 * iterator.hpp
 * 
 * 
 * Create Date : 2019-11-16 18:45:45
 * Copyright (c) 2019 Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef ITERATOR_HPP__
#define ITERATOR_HPP__

namespace dp {

template<class T> class Aggregate;

template<class T>
class Iterator {
  public:
    Iterator(Aggregate<T> *aggregate) : aggregate(aggregate), index(0) {}
    bool hasNext() { return index < aggregate->getLength(); };
    T next() {
        T next_member = aggregate->getAt(index);
        index++;
        return next_member;
    };

  private:
    Aggregate<T> *aggregate;
    int index;
};

}

#endif // ITERATOR_HPP__
