#include "training/gpu_backend.h"

#include <stdexcept>

namespace chiron::gpu {

#ifdef CHIRON_ENABLE_CUDA

void train_batch_cuda(const std::vector<TrainingExample>& batch, nnue::Network& network,
                      const Trainer::Config& config);

bool is_available() {
    return true;
}

void train_batch(const std::vector<TrainingExample>& batch, nnue::Network& network,
                 const Trainer::Config& config) {
    train_batch_cuda(batch, network, config);
}

#else

bool is_available() {
    return false;
}

void train_batch(const std::vector<TrainingExample>&, nnue::Network&, const Trainer::Config&) {
    throw std::runtime_error(
        "Chiron was built without CUDA support. Reconfigure with -DCHIRON_ENABLE_CUDA=ON to enable GPU training.");
}

#endif

}  // namespace chiron::gpu
