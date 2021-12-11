/*
 * test_iterator.cpp
 *
 * Create Date : 2019-11-17 01:17:04
 * Copyright (c) 2019 %author% <%email%>
 */

#include <iostream>

#include "aggregate.hpp"
#include "book.hpp"

int main() {
    dp::Aggregate<dp::Book> book_shelf(4);
    book_shelf.append(dp::Book("Around the World in 80 Days"));
    book_shelf.append(dp::Book("Bible"));
    book_shelf.append(dp::Book("Cinderella"));
    book_shelf.append(dp::Book("Daddy-Long-Legs"));

    dp::Iterator<dp::Book> iterator = book_shelf.iterator();

    while (iterator.hasNext()) {
        dp::Book book = iterator.next();
        std::cout << book.getName() << std::endl;
    }
}
