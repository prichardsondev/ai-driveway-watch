#include "wildlifeDetector.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: wildlife_ncnn_benchmark PARAM BIN IMAGE [RUNS]\n";
        return 2;
    }
    const int runs = argc > 4 ? std::max(1, std::atoi(argv[4])) : 10;
    const cv::Mat image = cv::imread(argv[3]);
    if (image.empty()) {
        std::cerr << "Could not read the benchmark image\n";
        return 2;
    }

    WildlifeDetector detector;
    if (detector.load(argv[1], argv[2]) != 0) {
        std::cerr << "Could not load the wildlife model\n";
        return 2;
    }

    std::vector<double> timings;
    std::vector<Object> animals;
    timings.reserve(runs);
    for (int index = 0; index < runs + 1; ++index) {
        const auto started = std::chrono::steady_clock::now();
        if (detector.detect(image, animals, 0.10f, 0.50f) != 0) {
            std::cerr << "Wildlife inference failed\n";
            return 2;
        }
        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        if (index > 0) {
            timings.push_back(milliseconds);
        }
    }

    const double average =
        std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    std::cout << std::fixed << std::setprecision(1)
              << "average_ms=" << average
              << " capacity_fps=" << 1000.0 / average
              << " animals=" << animals.size();
    if (!animals.empty()) {
        std::cout << " best_confidence=" << std::setprecision(3)
                  << animals.front().prob;
    }
    std::cout << '\n';
    return 0;
}
