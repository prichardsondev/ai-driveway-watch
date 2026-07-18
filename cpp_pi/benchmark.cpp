#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static cv::Mat letterbox(const cv::Mat& frame) {
    constexpr int size = 640;
    const double ratio = std::min(size / static_cast<double>(frame.rows),
                                  size / static_cast<double>(frame.cols));
    cv::Mat resized, rgb, padded(size, size, CV_32FC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized,
               cv::Size(static_cast<int>(frame.cols * ratio), static_cast<int>(frame.rows * ratio)));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3);
    rgb.copyTo(padded(cv::Rect(0, 0, rgb.cols, rgb.rows)));
    return cv::dnn::blobFromImage(padded);
}

int main(int argc, char** argv) {
    const char* camera_url = std::getenv("CAMERA_URL");
    if (!camera_url || argc < 2) {
        std::cerr << "usage: CAMERA_URL=... driveway_benchmark MODEL_PATH [seconds]\n";
        return 2;
    }
    const int duration_seconds = argc >= 3 ? std::max(5, std::atoi(argv[2])) : 30;
    cv::setNumThreads(4);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

    const std::vector<int> params = {
        cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000,
    };
    const auto open_started = Clock::now();
    cv::VideoCapture capture(cv::String(camera_url), cv::CAP_FFMPEG, params);
    const auto open_ms = std::chrono::duration<double, std::milli>(Clock::now() - open_started).count();
    if (!capture.isOpened()) {
        std::cerr << "camera_open=false open_ms=" << open_ms << "\n";
        return 3;
    }

    cv::Mat frame;
    std::vector<double> inference_ms;
    int frames = 0;
    const auto started = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - started).count() < duration_seconds) {
        if (!capture.read(frame)) break;
        ++frames;
        if (frames % 15 != 0) continue;
        cv::Mat blob = letterbox(frame);
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        const auto infer_started = Clock::now();
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        inference_ms.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - infer_started).count());
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const double average = inference_ms.empty() ? 0.0
        : std::accumulate(inference_ms.begin(), inference_ms.end(), 0.0) / inference_ms.size();
    std::cout << std::fixed << std::setprecision(1)
              << "opencv=" << CV_VERSION << "\n"
              << "threads=" << cv::getNumThreads() << "\n"
              << "resolution=" << frame.cols << "x" << frame.rows << "\n"
              << "open_ms=" << open_ms << "\n"
              << "frames=" << frames << "\n"
              << "capture_fps=" << frames / elapsed << "\n"
              << "inferences=" << inference_ms.size() << "\n"
              << "inference_ms_avg=" << average << "\n"
              << "inference_fps_capacity=" << (average > 0 ? 1000.0 / average : 0.0) << "\n";
}
