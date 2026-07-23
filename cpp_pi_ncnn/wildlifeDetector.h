#ifndef WILDLIFE_DETECTOR_H
#define WILDLIFE_DETECTOR_H

#include "yoloV8.h"

#include <filesystem>
#include <vector>

#include <net.h>
#include <opencv2/core/core.hpp>

class WildlifeDetector {
public:
    int load(const std::filesystem::path& param_path,
             const std::filesystem::path& model_path,
             int target_size = 640);
    int detect(const cv::Mat& bgr, std::vector<Object>& animals,
               float probability_threshold = 0.20f,
               float nms_threshold = 0.50f) const;

private:
    ncnn::Net detector_;
    int target_size_ = 640;
};

#endif
