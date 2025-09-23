#include "uci.h"

#include <iostream>

int main() {
    try {
        chiron::UCI uci;
        uci.loop();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

