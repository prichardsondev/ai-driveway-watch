#include "yoloV8.h"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}

int main(int argc, char** argv) {
    const char* env_url = std::getenv("CAMERA_URL");
    const std::string camera_url = argc > 1 ? argv[1] : (env_url ? env_url : "");
    const int duration_seconds = argc > 2 ? std::max(1, std::atoi(argv[2])) : 30;

    if (camera_url.empty()) {
        std::cerr << "Set CAMERA_URL or pass the stream URL as the first argument.\n";
        return 2;
    }

    cv::VideoCapture capture;
    const std::vector<int> capture_options = {
        cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
        cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000,
    };

    const auto open_start = Clock::now();
    if (!capture.open(camera_url, cv::CAP_FFMPEG, capture_options)) {
        std::cerr << "Could not open the camera stream.\n";
        return 3;
    }
    const double open_time_ms = elapsed_ms(open_start, Clock::now());

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double source_fps = capture.get(cv::CAP_PROP_FPS);

    std::atomic<bool> running{true};
    std::atomic<unsigned long> captured_frames{0};
    std::atomic<unsigned long> frame_generation{0};
    std::mutex frame_mutex;
    cv::Mat latest_frame;

    std::thread capture_thread([&] {
        cv::Mat frame;
        while (running.load(std::memory_order_relaxed)) {
            if (!capture.read(frame) || frame.empty()) {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame.copyTo(latest_frame);
            }
            captured_frames.fetch_add(1, std::memory_order_relaxed);
            frame_generation.fetch_add(1, std::memory_order_release);
        }
    });

    const auto first_frame_deadline = Clock::now() + std::chrono::seconds(12);
    while (frame_generation.load(std::memory_order_acquire) == 0 &&
           Clock::now() < first_frame_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (frame_generation.load(std::memory_order_acquire) == 0) {
        running.store(false);
        capture_thread.join();
        std::cerr << "Camera opened but no frames arrived.\n";
        return 4;
    }

    YoloV8 detector;
    if (detector.load(640) != 0) {
        running.store(false);
        capture_thread.join();
        std::cerr << "Could not load the YOLOv8-nano model.\n";
        return 5;
    }

    cv::Mat inference_frame;
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        latest_frame.copyTo(inference_frame);
    }
    std::vector<Object> objects;
    detector.detect(inference_frame, objects);  // warm-up

    const auto benchmark_start = Clock::now();
    const auto benchmark_end = benchmark_start + std::chrono::seconds(duration_seconds);
    const auto capture_count_start = captured_frames.load();
    unsigned long inference_count = 0;
    unsigned long useful_detections = 0;
    double total_inference_ms = 0.0;
    double fastest_inference_ms = 1e12;
    double slowest_inference_ms = 0.0;

    while (Clock::now() < benchmark_end) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame.copyTo(inference_frame);
        }

        objects.clear();
        const auto inference_start = Clock::now();
        detector.detect(inference_frame, objects);
        const double inference_ms = elapsed_ms(inference_start, Clock::now());

        total_inference_ms += inference_ms;
        fastest_inference_ms = std::min(fastest_inference_ms, inference_ms);
        slowest_inference_ms = std::max(slowest_inference_ms, inference_ms);
        ++inference_count;

        for (const auto& object : objects) {
            if (object.label == 0 || object.label == 1 || object.label == 2 ||
                object.label == 3 || object.label == 5 || object.label == 7) {
                ++useful_detections;
            }
        }
    }

    const double actual_seconds =
        std::chrono::duration<double>(Clock::now() - benchmark_start).count();
    running.store(false, std::memory_order_relaxed);
    capture_thread.join();
    capture.release();

    const auto frames_during_test = captured_frames.load() - capture_count_start;
    const double average_inference_ms = total_inference_ms / inference_count;

    std::cout << std::fixed << std::setprecision(1)
              << "stream=" << width << "x" << height << " @ " << source_fps << " fps\n"
              << "open_ms=" << open_time_ms << "\n"
              << "test_seconds=" << actual_seconds << "\n"
              << "capture_fps=" << frames_during_test / actual_seconds << "\n"
              << "inferences=" << inference_count << "\n"
              << "average_inference_ms=" << average_inference_ms << "\n"
              << "fastest_inference_ms=" << fastest_inference_ms << "\n"
              << "slowest_inference_ms=" << slowest_inference_ms << "\n"
              << "inference_capacity_fps=" << 1000.0 / average_inference_ms << "\n"
              << "person_or_vehicle_hits=" << useful_detections << "\n";

    return 0;
}
