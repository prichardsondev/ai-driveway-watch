#include "wildlifeDetector.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>

namespace {
constexpr int kAnimalLabel = 80;
constexpr float kPaddingValue = 114.0f;

float intersection_over_union(const Object& left, const Object& right) {
    const cv::Rect_<float> intersection = left.rect & right.rect;
    const float union_area =
        left.rect.area() + right.rect.area() - intersection.area();
    return union_area > 0.0f ? intersection.area() / union_area : 0.0f;
}

void apply_nms(std::vector<Object>& proposals, float threshold) {
    std::sort(proposals.begin(), proposals.end(),
              [](const Object& left, const Object& right) {
                  return left.prob > right.prob;
              });
    std::vector<Object> kept;
    kept.reserve(proposals.size());
    for (const auto& proposal : proposals) {
        const bool overlaps = std::any_of(
            kept.begin(), kept.end(), [&](const Object& existing) {
                return intersection_over_union(proposal, existing) > threshold;
            });
        if (!overlaps) {
            kept.push_back(proposal);
        }
    }
    proposals = std::move(kept);
}
}

int WildlifeDetector::load(const std::filesystem::path& param_path,
                           const std::filesystem::path& model_path,
                           int target_size) {
    detector_.clear();
    detector_.opt = ncnn::Option();
    detector_.opt.num_threads = 2;
    detector_.opt.use_packing_layout = true;
    target_size_ = std::max(32, target_size);

    const std::string param = param_path.string();
    const std::string model = model_path.string();
    const int param_result = detector_.load_param(param.c_str());
    if (param_result != 0) {
        return param_result;
    }
    return detector_.load_model(model.c_str());
}

int WildlifeDetector::detect(const cv::Mat& bgr, std::vector<Object>& animals,
                             float probability_threshold,
                             float nms_threshold) const {
    animals.clear();
    if (bgr.empty()) {
        return -1;
    }

    const int original_width = bgr.cols;
    const int original_height = bgr.rows;
    const float scale = std::min(
        static_cast<float>(target_size_) / original_width,
        static_cast<float>(target_size_) / original_height);
    const int resized_width =
        std::max(1, static_cast<int>(std::round(original_width * scale)));
    const int resized_height =
        std::max(1, static_cast<int>(std::round(original_height * scale)));
    const int width_padding = target_size_ - resized_width;
    const int height_padding = target_size_ - resized_height;
    const int left_padding = width_padding / 2;
    const int top_padding = height_padding / 2;

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
        original_width, original_height, resized_width, resized_height);
    ncnn::Mat input;
    ncnn::copy_make_border(
        resized, input,
        top_padding, height_padding - top_padding,
        left_padding, width_padding - left_padding,
        ncnn::BORDER_CONSTANT, kPaddingValue);
    const float normalization[] = {
        1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(nullptr, normalization);

    ncnn::Extractor extractor = detector_.create_extractor();
    if (extractor.input("in0", input) != 0) {
        return -2;
    }
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0 || output.h < 7) {
        return -3;
    }

    const float* center_x = output.row(0);
    const float* center_y = output.row(1);
    const float* width = output.row(2);
    const float* height = output.row(3);
    const float* animal_scores = output.row(4);
    for (int index = 0; index < output.w; ++index) {
        const float confidence = animal_scores[index];
        if (confidence < probability_threshold) {
            continue;
        }

        float x0 = center_x[index] - width[index] * 0.5f;
        float y0 = center_y[index] - height[index] * 0.5f;
        float x1 = center_x[index] + width[index] * 0.5f;
        float y1 = center_y[index] + height[index] * 0.5f;
        x0 = (x0 - left_padding) / scale;
        y0 = (y0 - top_padding) / scale;
        x1 = (x1 - left_padding) / scale;
        y1 = (y1 - top_padding) / scale;
        x0 = std::clamp(x0, 0.0f, static_cast<float>(original_width - 1));
        y0 = std::clamp(y0, 0.0f, static_cast<float>(original_height - 1));
        x1 = std::clamp(x1, 0.0f, static_cast<float>(original_width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(original_height - 1));
        if (x1 <= x0 || y1 <= y0) {
            continue;
        }
        animals.push_back({
            cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0),
            kAnimalLabel,
            confidence});
    }
    apply_nms(animals, nms_threshold);
    return 0;
}
