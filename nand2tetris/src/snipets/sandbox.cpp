#include <bits/stdc++.h>


constexpr double square(double a) { return a * a; }

int main() {
    constexpr double a = 7.7;
    constexpr double max1 = 1.4 * square(a);
    const double b = 7.8;
    const double max2 = 1.4 * square(b);
}
