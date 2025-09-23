#include <gtest/gtest.h>

#include <filesystem>

#include "training/trainer.h"

namespace chiron {

TEST(Training, SaveLoadRoundTrip) {
    TrainingExample example{"8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", 200};

    ParameterSet parameters;
    Trainer trainer({0.1, 0.0});
    trainer.train_batch({example}, parameters);
    int before = trainer.evaluate_example(example, parameters);

    std::filesystem::path temp = std::filesystem::temp_directory_path() / "chiron-training.nnue";
    parameters.save(temp.string());

    ParameterSet reloaded;
    reloaded.load(temp.string());
    int after = trainer.evaluate_example(example, reloaded);

    std::filesystem::remove(temp);
    EXPECT_EQ(before, after);
}

}  // namespace chiron

