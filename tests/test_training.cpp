#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "training/pgn_importer.h"
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

TEST(Training, PgnImporterOrientsTargets) {
    const char* pgn = R"([Event "Test"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Qh5 Ke7 3. Qxe5# 1-0
)";

    std::filesystem::path temp = std::filesystem::temp_directory_path() / "chiron-import-test.pgn";
    {
        std::ofstream out(temp);
        ASSERT_TRUE(out.good());
        out << pgn;
    }

    PgnImporter importer;
    std::vector<TrainingExample> examples = importer.import_file(temp.string());
    std::filesystem::remove(temp);

    ASSERT_GE(examples.size(), 2u);
    EXPECT_EQ(examples[0].target_cp, 1000);
    EXPECT_EQ(examples[1].target_cp, -1000);
}

}  // namespace chiron

