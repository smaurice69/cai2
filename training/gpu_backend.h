#pragma once

#include <vector>

#include "nnue/network.h"
#include "training/trainer.h"

namespace chiron::gpu {

bool is_available();
void train_batch(const std::vector<TrainingExample>& batch, nnue::Network& network,
                 const Trainer::Config& config);

}  // namespace chiron::gpu
