#pragma once

#include <string>
#include <vector>

namespace chiron {

struct TeacherConfig {
    std::string engine_path;
    int depth = 20;
    int threads = 1;
};

/**
 * @brief Offline annotator that queries an external UCI engine for evaluations.
 */
class TeacherEngine {
   public:
    explicit TeacherEngine(TeacherConfig config);

    std::vector<int> evaluate(const std::vector<std::string>& fens) const;
    int evaluate_single(const std::string& fen) const;

   private:
    TeacherConfig config_;
};

}  // namespace chiron

