/*
 * hand.hpp
 * 
 * 
 * Create Date : 2020-03-01 11:45:28
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef HAND_HPP__
#define HAND_HPP__

#include <vector>
#include <string>


namespace dp {

enum class HandValue {
    GUU = 0,
    CHO = 1,
    PAA = 2
};



class Hand {
  public:
    constexpr static HandValue hand[3] {HandValue::GUU, HandValue::CHO, HandValue::PAA};
    Hand(HandValue hand) : handvalue(hand) {}

    static Hand getHand(int handvalue) {
        return Hand(static_cast<HandValue>(handvalue));
    }

    bool isStrongerThan(Hand h) {
        return fight(h) == 1;
    }

    bool isWeakerThan(Hand h) {
        return fight(h) == -1;
    }

    std::string toString() {
        switch (handvalue) {
        case HandValue::GUU:
            return "guu";
        case HandValue::CHO:
            return "choki";
        case HandValue::PAA:
            return "paa";
        }
    }

  private:
    int fight(Hand h) {
        int opp = static_cast<int>(h.handvalue);
        int own = static_cast<int>(handvalue);
        if (opp == own) {
            return 0;
        } else if ((own + 1) % 3 == opp)  {
            return 1;
        } else {
            return -1;
        }
    }

    HandValue handvalue;
};

}

#endif // HAND_HPP__
