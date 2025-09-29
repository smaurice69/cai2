#pragma once

#include <cstddef>
#include <vector>

#include "training/trainer.h"

namespace chiron {

struct DatasetEvaluationResult {
    double accuracy = 0.0;
    double pseudo_elo = 0.0;
    std::size_t samples = 0;
};

DatasetEvaluationResult evaluate_dataset_performance(const std::vector<TrainingExample>& data,
                                                     const ParameterSet& parameters,
                                                     const Trainer& trainer,
                                                     std::size_t max_samples = 4096);

}  // namespace chiron

