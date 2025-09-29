#include "training/training_metrics.h"

#include <algorithm>
#include <cmath>

namespace chiron {

DatasetEvaluationResult evaluate_dataset_performance(const std::vector<TrainingExample>& data,
                                                     const ParameterSet& parameters,
                                                     const Trainer& trainer,
                                                     std::size_t max_samples) {
    DatasetEvaluationResult result;
    if (data.empty() || max_samples == 0) {
        return result;
    }

    std::size_t sample_count = std::min<std::size_t>(max_samples, data.size());
    double total_score = 0.0;
    double step = static_cast<double>(data.size()) / static_cast<double>(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) {
        std::size_t index = static_cast<std::size_t>(i * step);
        if (index >= data.size()) {
            index = data.size() - 1;
        }
        const TrainingExample& example = data[index];
        int predicted_cp = trainer.evaluate_example(example, parameters);
        double predicted_prob = 1.0 / (1.0 + std::exp(-static_cast<double>(predicted_cp) / 400.0));
        double actual_prob = 0.5;
        if (example.target_cp > 50) {
            actual_prob = 1.0;
        } else if (example.target_cp < -50) {
            actual_prob = 0.0;
        }
        total_score += 1.0 - std::fabs(predicted_prob - actual_prob);
    }

    result.samples = sample_count;
    result.accuracy = total_score / static_cast<double>(sample_count);
    double clipped = std::clamp(result.accuracy, 0.01, 0.99);
    result.pseudo_elo = 400.0 * std::log10(clipped / (1.0 - clipped));
    return result;
}

}  // namespace chiron

