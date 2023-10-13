// hello_world.cpp
#include <iostream>
#include <tuple>

int main() {
    auto [x, y] = std::make_tuple("Hello, ", "World!");
    std::cout << x << y << std::endl;
    return 0;
}