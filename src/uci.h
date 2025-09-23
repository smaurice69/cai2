#pragma once

#include <string>

#include "board.h"
#include "search.h"

namespace chiron {

/**
 * @brief Minimal UCI protocol front-end driving the search engine.
 */
class UCI {
   public:
    UCI();
    void loop();

   private:
    void handle_position(const std::string& command);
    void handle_go(const std::string& command);
    Move parse_move(const std::string& token);

    Board board_;
    Search search_;
};

}  // namespace chiron

