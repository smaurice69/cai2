#pragma once

#include <string>
#include <vector>

#include "training/trainer.h"

namespace chiron {

/**
 * @brief Utility for converting PGN databases into training examples.
 */
class PgnImporter {
   public:
    std::vector<TrainingExample> import_file(const std::string& path, bool include_draws = true) const;
    void write_dataset(const std::string& pgn_path, const std::string& output_path, bool include_draws = true) const;

   private:
    static int result_to_target(const std::string& result_tag);
};

}  // namespace chiron

