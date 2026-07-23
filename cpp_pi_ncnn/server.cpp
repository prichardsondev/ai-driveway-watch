#include "yoloV8.h"
#include "wildlifeDetector.h"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::atomic<bool> running{true};

struct ServiceConfig {
    std::vector<cv::Point2f> driveway_zone = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    double event_cooldown_seconds = 60.0;
    double event_clear_seconds = 5.0;
    float confidence_threshold = 0.40f;
    float person_confidence_threshold = 0.25f;
    float animal_confidence_threshold = 0.35f;
    bool animal_alerts_enabled = true;
    bool general_events_enabled = true;
    float wildlife_confidence_threshold = 0.20f;
    float nms_threshold = 0.50f;
    double detection_fps = 5.0;
    bool wildlife_model_enabled = false;
    double wildlife_detection_fps = 1.0;
    std::filesystem::path wildlife_param_path =
        "wildlife-model/model.ncnn.param";
    std::filesystem::path wildlife_model_path =
        "wildlife-model/model.ncnn.bin";
    bool species_classifier_enabled = false;
    bool species_classifier_required = true;
    int species_classifier_port = 8765;
    float species_confidence_threshold = 0.65f;
    double species_classifier_timeout_seconds = 8.0;
    double species_cache_seconds = 8.0;
    bool wildlife_test_captures_enabled = false;
    double wildlife_test_cooldown_seconds = 30.0;
    std::size_t wildlife_test_snapshot_retention = 200;
    std::filesystem::path manual_storage_root =
        "/mnt/driveway-recordings";
    double manual_recording_seconds = 30.0;
    double manual_storage_max_gb = 200.0;
    double manual_storage_min_free_gb = 10.0;
    std::string camera_url;
    double reconnect_delay_seconds = 2.0;
    std::filesystem::path output_dir = "runtime";
    std::size_t snapshot_retention = 200;
    std::string ntfy_server;
    std::string ntfy_topic;
    std::vector<cv::Point2f> mailbox_zone;
    double mailbox_dwell_seconds = 2.5;
    double mailbox_lost_seconds = 1.0;
    double mailbox_cooldown_seconds = 300.0;
    double mailbox_max_movement = 0.035;
    std::vector<cv::Point2f> road_zone;
    double road_track_lost_seconds = 0.8;
    double road_track_match_distance = 0.20;
    std::size_t road_snapshot_retention = 100000;
    std::string driveway_label = "Driveway arrival";
    std::string mailbox_label = "Mailbox stop";
    std::string road_label = "Road archive";
};

struct ZoneSettings {
    std::vector<cv::Point2f> driveway;
    std::vector<cv::Point2f> mailbox;
    std::vector<cv::Point2f> road;
    std::string driveway_label;
    std::string mailbox_label;
    std::string road_label;
};

struct EventInfo {
    std::string id;
    std::string timestamp;
    std::string class_name;
    float confidence = 0.0f;
    std::string snapshot;
};

ServiceConfig config;
std::mutex zones_mutex;
ZoneSettings default_zones;

struct SharedState {
    std::mutex frame_mutex;
    cv::Mat latest_frame;
    std::vector<unsigned char> latest_jpeg;
    std::vector<unsigned char> latest_clean_jpeg;
    unsigned long frame_generation = 0;
    unsigned long jpeg_generation = 0;

    std::mutex status_mutex;
    bool camera_connected = false;
    double camera_fps = 0.0;
    double inference_ms = 0.0;
    double average_inference_ms = 0.0;
    bool wildlife_model_ready = false;
    double wildlife_inference_ms = 0.0;
    unsigned long wildlife_inference_count = 0;
    bool species_classifier_ready = false;
    double species_inference_ms = 0.0;
    unsigned long species_inference_count = 0;
    std::string last_species;
    double last_species_confidence = 0.0;
    std::string species_classifier_error;
    unsigned long captured_frames = 0;
    unsigned long inference_count = 0;
    int relevant_objects = 0;
    double best_person_confidence = 0.0;
    bool person_in_zone = false;
    bool mailbox_vehicle_present = false;
    double mailbox_stationary_seconds = 0.0;
    unsigned long event_count = 0;
    unsigned long road_event_count = 0;
    unsigned long wildlife_test_event_count = 0;
    unsigned long notification_count = 0;
    std::deque<EventInfo> recent_events;
    std::deque<EventInfo> recent_road_events;
    std::deque<EventInfo> recent_wildlife_test_events;
    std::string last_error;
    std::string notification_error;
};

SharedState state;
std::atomic<unsigned int> clean_stream_clients{0};
std::atomic<unsigned long> manual_capture_sequence{0};
std::mutex manual_capture_mutex;
std::mutex manual_recording_mutex;
bool manual_recording_active = false;
pid_t manual_recording_pid = -1;
Clock::time_point manual_recording_started;
std::string manual_recording_filename;
std::string manual_recording_error;
std::mutex events_mutex;
std::mutex notification_mutex;
std::condition_variable notification_ready;
std::deque<EventInfo> notification_queue;

void stop_handler(int) {
    running.store(false);
}

bool send_all(int socket_fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::size_t sent = 0;
    while (sent < size && running.load()) {
        const ssize_t result =
            ::send(socket_fd, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (result <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(result);
    }
    return sent == size;
}

bool send_text(int socket_fd, const std::string& content_type,
               const std::string& body, const std::string& status = "200 OK") {
    std::ostringstream headers;
    headers << "HTTP/1.1 " << status << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Cache-Control: no-store\r\n"
            << "Connection: close\r\n\r\n";
    const std::string header_text = headers.str();
    return send_all(socket_fd, header_text.data(), header_text.size()) &&
           send_all(socket_fd, body.data(), body.size());
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            result.push_back('\\');
        }
        if (ch == '\n' || ch == '\r') {
            result.push_back(' ');
        } else {
            result.push_back(ch);
        }
    }
    return result;
}

double environment_number(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value ? parsed : fallback;
}

bool environment_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (normalized == "1" || normalized == "true" ||
        normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" ||
        normalized == "no" || normalized == "off") {
        return false;
    }
    return fallback;
}

struct SpeciesResult {
    bool available = false;
    std::string label;
    float confidence = 0.0f;
    std::string error;
};

std::string species_slug(const std::string& label) {
    std::string slug;
    for (const unsigned char character : label) {
        if (std::isalnum(character)) {
            slug.push_back(static_cast<char>(std::tolower(character)));
        } else if ((character == ' ' || character == '-' || character == '_') &&
                   !slug.empty() && slug.back() != '_') {
            slug.push_back('_');
        }
    }
    while (!slug.empty() && slug.back() == '_') {
        slug.pop_back();
    }
    return slug.empty() ? "wildlife" : slug;
}

bool rejected_species_label(const std::string& label) {
    const std::string slug = species_slug(label);
    return slug == "no_species" || slug == "human";
}

SpeciesResult classify_species_crop(const cv::Mat& crop) {
    SpeciesResult result;
    if (crop.empty()) {
        result.error = "empty animal crop";
        return result;
    }
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(
            ".jpg", crop, jpeg, {cv::IMWRITE_JPEG_QUALITY, 90}) ||
        jpeg.empty()) {
        result.error = "could not encode animal crop";
        return result;
    }

    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        result.error = "could not create classifier socket";
        return result;
    }
    const auto timeout_microseconds = static_cast<long>(
        std::max(0.1, config.species_classifier_timeout_seconds) * 1e6);
    timeval timeout{};
    timeout.tv_sec = timeout_microseconds / 1000000;
    timeout.tv_usec = timeout_microseconds % 1000000;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port =
        htons(static_cast<uint16_t>(config.species_classifier_port));
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (connect(socket_fd, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
        close(socket_fd);
        result.error = "regional classifier is not ready";
        return result;
    }

    std::ostringstream request;
    request << "POST /classify HTTP/1.1\r\n"
            << "Host: 127.0.0.1\r\n"
            << "Content-Type: image/jpeg\r\n"
            << "Content-Length: " << jpeg.size() << "\r\n"
            << "Connection: close\r\n\r\n";
    const std::string headers = request.str();
    if (!send_all(socket_fd, headers.data(), headers.size()) ||
        !send_all(socket_fd, jpeg.data(), jpeg.size())) {
        close(socket_fd);
        result.error = "could not send crop to regional classifier";
        return result;
    }
    shutdown(socket_fd, SHUT_WR);

    std::string response;
    std::array<char, 1024> buffer{};
    while (response.size() < 16384) {
        const ssize_t received = recv(
            socket_fd, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            break;
        }
        response.append(buffer.data(), static_cast<std::size_t>(received));
    }
    close(socket_fd);
    if (response.rfind("HTTP/1.0 200", 0) != 0 &&
        response.rfind("HTTP/1.1 200", 0) != 0) {
        result.error = "regional classifier rejected the crop";
        return result;
    }
    const auto body_offset = response.find("\r\n\r\n");
    if (body_offset == std::string::npos) {
        result.error = "invalid regional classifier response";
        return result;
    }
    const std::string body = response.substr(body_offset + 4);
    const auto separator = body.find('\t');
    if (separator == std::string::npos) {
        result.error = "invalid regional classifier result";
        return result;
    }
    result.label = body.substr(0, separator);
    try {
        result.confidence = std::stof(body.substr(separator + 1));
    } catch (...) {
        result.error = "invalid regional classifier confidence";
        result.label.clear();
        return result;
    }
    result.available = !result.label.empty() &&
        std::isfinite(result.confidence);
    if (!result.available) {
        result.error = "empty regional classifier result";
    }
    return result;
}

bool valid_ntfy_topic(const std::string& topic) {
    return !topic.empty() && std::all_of(topic.begin(), topic.end(), [](unsigned char ch) {
        return std::isalnum(ch) || ch == '-' || ch == '_';
    });
}

bool ntfy_enabled() {
    return (config.ntfy_server.rfind("https://", 0) == 0 ||
            config.ntfy_server.rfind("http://", 0) == 0) &&
           valid_ntfy_topic(config.ntfy_topic);
}

void queue_notification(const EventInfo& event) {
    if (!ntfy_enabled()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(notification_mutex);
        if (notification_queue.size() >= 20) {
            notification_queue.pop_front();
        }
        notification_queue.push_back(event);
    }
    notification_ready.notify_one();
}

void send_ntfy(const EventInfo& event) {
    std::string server = config.ntfy_server;
    while (!server.empty() && server.back() == '/') {
        server.pop_back();
    }
    const std::string url = server + "/" + config.ntfy_topic;
    std::string display_class = event.class_name;
    std::replace(display_class.begin(), display_class.end(), '_', ' ');
    if (!display_class.empty()) {
        display_class.front() = static_cast<char>(std::toupper(display_class.front()));
    }
    const bool mailbox_event = event.class_name == "mailbox_vehicle";
    const bool animal_event = event.class_name.rfind("animal_", 0) == 0;
    std::string animal_name = animal_event ? event.class_name.substr(7) : "";
    std::replace(animal_name.begin(), animal_name.end(), '_', ' ');
    if (!animal_name.empty()) {
        animal_name.front() = static_cast<char>(std::toupper(animal_name.front()));
    }
    const std::string title = mailbox_event
        ? "Title: Driveway Watch — Mailbox"
        : (animal_event ? "Title: Driveway Watch — Animal"
                        : "Title: Driveway Watch — " + display_class);
    const std::string tags = mailbox_event ? "Tags: mailbox_with_mail"
        : (animal_event ? "Tags: paw_prints"
                        : (event.class_name == "person"
                               ? "Tags: bust_in_silhouette" : "Tags: car"));
    std::ostringstream body;
    if (mailbox_event) {
        body << "Vehicle stopped near the mailbox (";
    } else if (animal_event) {
        body << (animal_name.empty() ? "Animal" : animal_name)
             << " detected inside a monitored boundary (";
    } else {
        body << display_class << " detected in the driveway (";
    }
    body << std::fixed << std::setprecision(0) << event.confidence * 100.0f
         << "% confidence).";
    const std::string message = body.str();

    const pid_t child = fork();
    if (child == 0) {
        execl("/usr/bin/curl", "curl", "--silent", "--show-error", "--fail",
              "--max-time", "10", "--retry", "2", "-H", title.c_str(),
              "-H", "Priority: default", "-H", tags.c_str(), "--data-binary",
              message.c_str(), url.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    bool delivered = false;
    if (child > 0) {
        int status = 0;
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
        delivered = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    std::lock_guard<std::mutex> lock(state.status_mutex);
    if (delivered) {
        ++state.notification_count;
        state.notification_error.clear();
    } else {
        state.notification_error = "ntfy delivery failed; it will try again on the next event";
    }
}

void notification_loop() {
    while (running.load()) {
        EventInfo event;
        {
            std::unique_lock<std::mutex> lock(notification_mutex);
            notification_ready.wait_for(lock, std::chrono::seconds(1), [] {
                return !running.load() || !notification_queue.empty();
            });
            if (!running.load()) {
                break;
            }
            if (notification_queue.empty()) {
                continue;
            }
            event = std::move(notification_queue.front());
            notification_queue.pop_front();
        }
        send_ntfy(event);
    }
}

std::vector<cv::Point2f> parse_zone(const char* value) {
    if (!value || !*value) {
        return config.driveway_zone;
    }
    std::vector<cv::Point2f> points;
    std::istringstream pairs(value);
    std::string pair;
    while (std::getline(pairs, pair, ';')) {
        const std::size_t comma = pair.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        try {
            const float x = std::stof(pair.substr(0, comma));
            const float y = std::stof(pair.substr(comma + 1));
            if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f) {
                points.emplace_back(x, y);
            }
        } catch (...) {
            points.clear();
            break;
        }
    }
    return points.size() >= 3 ? points : config.driveway_zone;
}

std::vector<cv::Point2f> parse_optional_zone(const char* value) {
    if (!value || !*value) {
        return {};
    }
    std::vector<cv::Point2f> points;
    std::istringstream pairs(value);
    std::string pair;
    while (std::getline(pairs, pair, ';')) {
        const std::size_t comma = pair.find(',');
        if (comma == std::string::npos) {
            return {};
        }
        try {
            const float x = std::stof(pair.substr(0, comma));
            const float y = std::stof(pair.substr(comma + 1));
            if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
                return {};
            }
            points.emplace_back(x, y);
        } catch (...) {
            return {};
        }
    }
    return points.size() >= 3 ? points : std::vector<cv::Point2f>{};
}

std::string clean_zone_label(const std::string& value, const std::string& fallback) {
    std::string cleaned;
    cleaned.reserve(std::min<std::size_t>(value.size(), 48));
    for (const unsigned char character : value) {
        if (character >= 32 && character != 127 && character != '\t' &&
            character != '\r' && character != '\n') {
            cleaned.push_back(static_cast<char>(character));
            if (cleaned.size() == 48) {
                break;
            }
        }
    }
    const auto first = cleaned.find_first_not_of(' ');
    if (first == std::string::npos) {
        return fallback;
    }
    const auto last = cleaned.find_last_not_of(' ');
    return cleaned.substr(first, last - first + 1);
}

std::string zone_points_text(const std::vector<cv::Point2f>& points) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(5);
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (index) {
            text << ';';
        }
        text << points[index].x << ',' << points[index].y;
    }
    return text.str();
}

bool same_zone(const std::vector<cv::Point2f>& left,
               const std::vector<cv::Point2f>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (cv::norm(left[index] - right[index]) > 0.00001f) {
            return false;
        }
    }
    return true;
}

ZoneSettings current_zones_locked() {
    return {config.driveway_zone, config.mailbox_zone, config.road_zone,
            config.driveway_label, config.mailbox_label, config.road_label};
}

void apply_zones_locked(const ZoneSettings& zones) {
    config.driveway_zone = zones.driveway;
    config.mailbox_zone = zones.mailbox;
    config.road_zone = zones.road;
    config.driveway_label = zones.driveway_label;
    config.mailbox_label = zones.mailbox_label;
    config.road_label = zones.road_label;
}

bool zones_are_default_locked() {
    const ZoneSettings current = current_zones_locked();
    return same_zone(current.driveway, default_zones.driveway) &&
           same_zone(current.mailbox, default_zones.mailbox) &&
           same_zone(current.road, default_zones.road) &&
           current.driveway_label == default_zones.driveway_label &&
           current.mailbox_label == default_zones.mailbox_label &&
           current.road_label == default_zones.road_label;
}

bool save_zones_locked() {
    const auto destination = config.output_dir / "zones.conf";
    const auto temporary = config.output_dir / "zones.conf.tmp";
    std::ofstream file(temporary, std::ios::trunc);
    if (!file) {
        return false;
    }
    const ZoneSettings zones = current_zones_locked();
    file << "driveway\t" << zones.driveway_label << '\t'
         << zone_points_text(zones.driveway) << '\n'
         << "mailbox\t" << zones.mailbox_label << '\t'
         << zone_points_text(zones.mailbox) << '\n'
         << "road\t" << zones.road_label << '\t'
         << zone_points_text(zones.road) << '\n';
    file.close();
    if (!file) {
        return false;
    }
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (!error) {
        return true;
    }
    std::filesystem::remove(destination, error);
    error.clear();
    std::filesystem::rename(temporary, destination, error);
    return !error;
}

void load_saved_zones() {
    std::ifstream file(config.output_dir / "zones.conf");
    if (!file) {
        return;
    }
    ZoneSettings loaded = default_zones;
    bool driveway_valid = false;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t first_tab = line.find('\t');
        const std::size_t second_tab = first_tab == std::string::npos
            ? std::string::npos : line.find('\t', first_tab + 1);
        if (first_tab == std::string::npos || second_tab == std::string::npos) {
            continue;
        }
        const std::string id = line.substr(0, first_tab);
        const std::string label = line.substr(first_tab + 1, second_tab - first_tab - 1);
        const std::string points_text = line.substr(second_tab + 1);
        const auto points = parse_optional_zone(points_text.c_str());
        if (points.size() < 3) {
            continue;
        }
        if (id == "driveway") {
            loaded.driveway = points;
            loaded.driveway_label = clean_zone_label(label, default_zones.driveway_label);
            driveway_valid = true;
        } else if (id == "mailbox") {
            loaded.mailbox = points;
            loaded.mailbox_label = clean_zone_label(label, default_zones.mailbox_label);
        } else if (id == "road") {
            loaded.road = points;
            loaded.road_label = clean_zone_label(label, default_zones.road_label);
        }
    }
    if (driveway_valid) {
        std::lock_guard<std::mutex> lock(zones_mutex);
        apply_zones_locked(loaded);
    }
}

std::string zone_points_json(const std::vector<cv::Point2f>& points) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(5) << '[';
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (index) {
            json << ',';
        }
        json << '[' << points[index].x << ',' << points[index].y << ']';
    }
    json << ']';
    return json.str();
}

std::string zones_json() {
    std::lock_guard<std::mutex> lock(zones_mutex);
    const ZoneSettings zones = current_zones_locked();
    std::ostringstream json;
    json << "{\"customized\":" << (zones_are_default_locked() ? "false" : "true")
         << ",\"zones\":["
         << "{\"id\":\"driveway\",\"label\":\"" << json_escape(zones.driveway_label)
         << "\",\"behavior\":\"Alerts for people and vehicles\",\"color\":\"#41bef7\",\"points\":"
         << zone_points_json(zones.driveway) << "},"
         << "{\"id\":\"mailbox\",\"label\":\"" << json_escape(zones.mailbox_label)
         << "\",\"behavior\":\"Alerts when a vehicle stops\",\"color\":\"#ffb900\",\"points\":"
         << zone_points_json(zones.mailbox) << "},"
         << "{\"id\":\"road\",\"label\":\"" << json_escape(zones.road_label)
         << "\",\"behavior\":\"Archives vehicles without alerts\",\"color\":\"#d250d2\",\"points\":"
         << zone_points_json(zones.road) << "}]}";
    return json.str();
}

int hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    return -1;
}

std::string url_decode(const std::string& value) {
    std::string decoded;
    decoded.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '+') {
            decoded.push_back(' ');
        } else if (value[index] == '%' && index + 2 < value.size()) {
            const int high = hex_value(value[index + 1]);
            const int low = hex_value(value[index + 2]);
            if (high >= 0 && low >= 0) {
                decoded.push_back(static_cast<char>((high << 4) | low));
                index += 2;
            } else {
                decoded.push_back(value[index]);
            }
        } else {
            decoded.push_back(value[index]);
        }
    }
    return decoded;
}

std::map<std::string, std::string> parse_form(const std::string& body) {
    std::map<std::string, std::string> values;
    std::istringstream fields(body);
    std::string field;
    while (std::getline(fields, field, '&')) {
        const std::size_t equals = field.find('=');
        if (equals != std::string::npos) {
            values[url_decode(field.substr(0, equals))] =
                url_decode(field.substr(equals + 1));
        }
    }
    return values;
}

bool update_zones_from_form(const std::string& body, std::string& error_message) {
    const auto values = parse_form(body);
    const auto driveway_value = values.find("driveway_points");
    const auto mailbox_value = values.find("mailbox_points");
    const auto road_value = values.find("road_points");
    if (driveway_value == values.end() || mailbox_value == values.end() ||
        road_value == values.end()) {
        error_message = "All three boundary types are required.";
        return false;
    }
    const auto driveway = parse_optional_zone(driveway_value->second.c_str());
    const auto mailbox = parse_optional_zone(mailbox_value->second.c_str());
    const auto road = parse_optional_zone(road_value->second.c_str());
    if (driveway.size() < 3 || mailbox.size() < 3 || road.size() < 3) {
        error_message = "Each boundary needs at least three points.";
        return false;
    }
    ZoneSettings updated{driveway, mailbox, road,
        clean_zone_label(values.count("driveway_label") ? values.at("driveway_label") : "",
                         "Driveway arrival"),
        clean_zone_label(values.count("mailbox_label") ? values.at("mailbox_label") : "",
                         "Mailbox stop"),
        clean_zone_label(values.count("road_label") ? values.at("road_label") : "",
                         "Road archive")};
    std::lock_guard<std::mutex> lock(zones_mutex);
    const ZoneSettings previous = current_zones_locked();
    apply_zones_locked(updated);
    if (!save_zones_locked()) {
        apply_zones_locked(previous);
        error_message = "The boundary file could not be saved.";
        return false;
    }
    return true;
}

bool restore_default_zones() {
    std::lock_guard<std::mutex> lock(zones_mutex);
    const ZoneSettings previous = current_zones_locked();
    apply_zones_locked(default_zones);
    if (!save_zones_locked()) {
        apply_zones_locked(previous);
        return false;
    }
    return true;
}

std::string utc_timestamp(std::string* filename_timestamp = nullptr) {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream display;
    display << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    if (filename_timestamp) {
        std::ostringstream filename;
        filename << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
        *filename_timestamp = filename.str();
    }
    return display.str();
}

std::filesystem::path manual_directory(const std::string& kind = "") {
    auto directory = config.manual_storage_root / "manual";
    return kind.empty() ? directory : directory / kind;
}

bool manual_storage_ready() {
    struct stat storage_status{};
    struct stat parent_status{};
    const auto root = config.manual_storage_root;
    if (root.empty() || root == root.root_path() ||
        ::stat(root.c_str(), &storage_status) != 0 ||
        ::stat(root.parent_path().c_str(), &parent_status) != 0) {
        return false;
    }
    return S_ISDIR(storage_status.st_mode) &&
           storage_status.st_dev != parent_status.st_dev;
}

std::string manual_capture_filename(const std::string& kind,
                                    const std::string& extension) {
    std::string timestamp;
    utc_timestamp(&timestamp);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    const auto sequence = ++manual_capture_sequence;
    return "manual-" + kind + "-" + timestamp + "-" +
           std::to_string(milliseconds % 1000) + "-" +
           std::to_string(sequence) + extension;
}

bool valid_manual_capture(const std::string& kind, const std::string& filename) {
    if ((kind != "snapshots" && kind != "videos") ||
        filename.empty() || filename.find('/') != std::string::npos ||
        filename.find("..") != std::string::npos ||
        filename.rfind("manual-", 0) != 0) {
        return false;
    }
    const auto extension = std::filesystem::path(filename).extension();
    return (kind == "snapshots" && extension == ".jpg") ||
           (kind == "videos" && extension == ".mp4");
}

void enforce_manual_retention() {
    if (!manual_storage_ready()) {
        return;
    }
    std::vector<std::filesystem::directory_entry> captures;
    std::uintmax_t total_bytes = 0;
    std::error_code error;
    for (const std::string kind : {"snapshots", "videos"}) {
        for (const auto& entry :
             std::filesystem::directory_iterator(manual_directory(kind), error)) {
            if (!entry.is_regular_file(error) || entry.is_symlink(error)) {
                continue;
            }
            const auto extension = entry.path().extension();
            if (extension != ".jpg" && extension != ".mp4") {
                continue;
            }
            total_bytes += entry.file_size(error);
            captures.push_back(entry);
        }
        error.clear();
    }
    std::sort(captures.begin(), captures.end(),
              [](const auto& left, const auto& right) {
                  return left.last_write_time() < right.last_write_time();
              });
    const std::uintmax_t max_bytes = static_cast<std::uintmax_t>(
        config.manual_storage_max_gb * 1024.0 * 1024.0 * 1024.0);
    const std::uintmax_t minimum_free = static_cast<std::uintmax_t>(
        config.manual_storage_min_free_gb * 1024.0 * 1024.0 * 1024.0);
    auto space = std::filesystem::space(config.manual_storage_root, error);
    if (error) {
        return;
    }
    std::uintmax_t available = space.available;
    for (const auto& capture : captures) {
        if (total_bytes <= max_bytes && available >= minimum_free) {
            break;
        }
        const auto size = capture.file_size(error);
        error.clear();
        if (std::filesystem::remove(capture.path(), error) && !error) {
            total_bytes = size <= total_bytes ? total_bytes - size : 0;
            available += size;
        }
        error.clear();
    }
}

int save_manual_snapshots(int requested_count, std::string& error_message) {
    if (!manual_storage_ready()) {
        error_message = "Recording SSD is not mounted.";
        return 0;
    }
    const int count =
        requested_count == 3 || requested_count == 5 ? requested_count : 1;
    std::lock_guard<std::mutex> capture_lock(manual_capture_mutex);
    std::error_code error;
    std::filesystem::create_directories(manual_directory("snapshots"), error);
    if (error) {
        error_message = "Snapshot directory is unavailable.";
        return 0;
    }

    int saved = 0;
    for (int index = 0; index < count; ++index) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> frame_lock(state.frame_mutex);
            if (!state.latest_frame.empty()) {
                state.latest_frame.copyTo(frame);
            }
        }
        if (frame.empty()) {
            error_message = "The current camera frame is not ready.";
            break;
        }
        std::vector<unsigned char> encoded;
        try {
            if (!cv::imencode(
                    ".jpg", frame, encoded,
                    {cv::IMWRITE_JPEG_QUALITY, 92})) {
                error_message = "The current camera frame could not be encoded.";
                break;
            }
        } catch (const cv::Exception&) {
            error_message = "The current camera frame could not be encoded.";
            break;
        }
        const auto final_path = manual_directory("snapshots") /
            manual_capture_filename("snapshot", ".jpg");
        std::ofstream output(final_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(encoded.data()),
                     static_cast<std::streamsize>(encoded.size()));
        output.close();
        std::error_code file_error;
        const bool saved_file = std::filesystem::is_regular_file(
            final_path, file_error) && !file_error &&
            std::filesystem::file_size(final_path, file_error) > 0 && !file_error;
        if (!output.good() || !saved_file) {
            std::error_code cleanup_error;
            std::filesystem::remove(final_path, cleanup_error);
            error_message = "The snapshot could not be written to the SSD.";
            break;
        }
        ++saved;
        if (index + 1 < count) {
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
        }
    }
    enforce_manual_retention();
    return saved;
}

bool start_manual_recording(std::string& error_message) {
    if (!manual_storage_ready()) {
        error_message = "Recording SSD is not mounted.";
        return false;
    }
    if (config.camera_url.empty()) {
        error_message = "Camera stream is unavailable.";
        return false;
    }
    std::lock_guard<std::mutex> recording_lock(manual_recording_mutex);
    if (manual_recording_active) {
        error_message = "A recording is already active.";
        return false;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(
        manual_directory("videos"), filesystem_error);
    if (filesystem_error) {
        error_message = "Video directory is unavailable.";
        return false;
    }
    const std::string filename =
        manual_capture_filename("recording", ".mp4");
    const auto final_path = manual_directory("videos") / filename;
    auto temporary_path = final_path;
    temporary_path += ".part";
    const std::string duration =
        std::to_string(static_cast<int>(std::lround(
            config.manual_recording_seconds)));
    const std::string output = temporary_path.string();

    const pid_t child = fork();
    if (child == 0) {
        const int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDOUT_FILENO);
            dup2(null_fd, STDERR_FILENO);
            close(null_fd);
        }
        execl("/usr/bin/ffmpeg", "ffmpeg",
              "-nostdin", "-hide_banner", "-loglevel", "error",
              "-rtsp_transport", "tcp", "-i", config.camera_url.c_str(),
              "-t", duration.c_str(), "-map", "0:v:0",
              "-c:v", "copy", "-an", "-movflags", "+faststart",
              "-f", "mp4", "-y", output.c_str(),
              static_cast<char*>(nullptr));
        _exit(127);
    }
    if (child < 0) {
        error_message = "The recorder process could not start.";
        return false;
    }

    manual_recording_active = true;
    manual_recording_pid = child;
    manual_recording_started = Clock::now();
    manual_recording_filename = filename;
    manual_recording_error.clear();
    std::thread([child, temporary_path, final_path]() {
        int status = 0;
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
        std::error_code error;
        const bool has_video =
            std::filesystem::exists(temporary_path, error) &&
            std::filesystem::file_size(temporary_path, error) > 0;
        bool saved_video = false;
        if (has_video) {
            std::filesystem::rename(temporary_path, final_path, error);
            saved_video = !error;
        }
        if (!saved_video) {
            error.clear();
            std::filesystem::remove(temporary_path, error);
        } else {
            std::lock_guard<std::mutex> capture_lock(manual_capture_mutex);
            enforce_manual_retention();
        }
        std::lock_guard<std::mutex> lock(manual_recording_mutex);
        if (manual_recording_pid == child) {
            manual_recording_active = false;
            manual_recording_pid = -1;
            if (!saved_video) {
                manual_recording_error = "Recording failed before a video was saved.";
            } else {
                manual_recording_error.clear();
            }
        }
    }).detach();
    return true;
}

bool stop_manual_recording(std::string& error_message) {
    std::lock_guard<std::mutex> lock(manual_recording_mutex);
    if (!manual_recording_active || manual_recording_pid <= 0) {
        error_message = "No recording is active.";
        return false;
    }
    if (kill(manual_recording_pid, SIGINT) != 0) {
        error_message = "The recorder could not be stopped.";
        return false;
    }
    return true;
}

std::string manual_captures_json() {
    std::lock_guard<std::mutex> capture_lock(manual_capture_mutex);
    struct ManualFile {
        std::string kind;
        std::string filename;
        std::uintmax_t size = 0;
        std::filesystem::file_time_type modified;
    };
    std::vector<ManualFile> files;
    std::error_code error;
    if (manual_storage_ready()) {
        for (const std::string kind : {"snapshots", "videos"}) {
            for (const auto& entry :
                 std::filesystem::directory_iterator(manual_directory(kind), error)) {
                if (!entry.is_regular_file(error) || entry.is_symlink(error) ||
                    !valid_manual_capture(kind, entry.path().filename().string())) {
                    continue;
                }
                files.push_back({
                    kind, entry.path().filename().string(),
                    entry.file_size(error), entry.last_write_time(error)});
            }
            error.clear();
        }
    }
    std::sort(files.begin(), files.end(),
              [](const ManualFile& left, const ManualFile& right) {
                  return left.modified > right.modified;
              });
    if (files.size() > 100) {
        files.resize(100);
    }
    std::ostringstream json;
    json << '[';
    for (std::size_t index = 0; index < files.size(); ++index) {
        if (index) {
            json << ',';
        }
        const auto& file = files[index];
        const auto system_time =
            std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                file.modified - std::filesystem::file_time_type::clock::now() +
                std::chrono::system_clock::now());
        const auto epoch_millis =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                system_time.time_since_epoch()).count();
        json << "{\"kind\":\"" << file.kind
             << "\",\"name\":\"" << json_escape(file.filename)
             << "\",\"size_bytes\":" << file.size
             << ",\"timestamp_ms\":" << epoch_millis
             << ",\"url\":\"/manual/" << file.kind << "/"
             << json_escape(file.filename)
             << "\",\"delete_path\":\"/api/manual-captures/"
             << file.kind << "/" << json_escape(file.filename) << "\"}";
    }
    json << ']';
    return json.str();
}

bool delete_manual_capture(const std::string& kind,
                           const std::string& filename) {
    if (!manual_storage_ready() || !valid_manual_capture(kind, filename)) {
        return false;
    }
    std::lock_guard<std::mutex> capture_lock(manual_capture_mutex);
    std::error_code error;
    return std::filesystem::remove(
        manual_directory(kind) / filename, error) && !error;
}

void enforce_snapshot_retention(const std::filesystem::path& events_dir,
                                std::size_t retention) {
    std::vector<std::filesystem::directory_entry> snapshots;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(events_dir, error)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            snapshots.push_back(entry);
        }
    }
    if (snapshots.size() <= retention) {
        return;
    }
    std::sort(snapshots.begin(), snapshots.end(), [](const auto& left, const auto& right) {
        return left.last_write_time() < right.last_write_time();
    });
    for (std::size_t index = 0;
         index < snapshots.size() - retention; ++index) {
        std::filesystem::remove(snapshots[index].path(), error);
    }
}

EventInfo create_archived_event(const cv::Mat& frame, const std::string& class_name,
                                float confidence, unsigned long sequence,
                                const std::string& archive_directory,
                                const std::string& log_filename,
                                std::size_t retention) {
    std::lock_guard<std::mutex> events_lock(events_mutex);
    const auto events_dir = config.output_dir / archive_directory;
    std::filesystem::create_directories(events_dir);
    std::string filename_timestamp;
    const std::string timestamp = utc_timestamp(&filename_timestamp);
    const auto epoch_millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string event_id = filename_timestamp + "-" +
        std::to_string(epoch_millis % 1000000) + "-" + std::to_string(sequence);
    const std::string snapshot = event_id + "-" + class_name + ".jpg";
    cv::imwrite((events_dir / snapshot).string(), frame);

    const auto log_path = config.output_dir / log_filename;
    const bool new_log = !std::filesystem::exists(log_path);
    std::ofstream log(log_path, std::ios::app);
    if (new_log) {
        log << "event_id,timestamp,class_name,confidence,snapshot\n";
    }
    log << event_id << ',' << timestamp << ',' << class_name << ','
        << std::fixed << std::setprecision(4) << confidence << ',' << snapshot << '\n';
    if (archive_directory != "road_events" || sequence % 100 == 0) {
        enforce_snapshot_retention(events_dir, retention);
    }
    return {event_id, timestamp, class_name, confidence, snapshot};
}

EventInfo create_event(const cv::Mat& frame, const std::string& class_name,
                       float confidence, unsigned long sequence) {
    return create_archived_event(frame, class_name, confidence, sequence,
                                 "events", "events.csv", config.snapshot_retention);
}

EventInfo create_road_event(const cv::Mat& frame, const std::string& class_name,
                            float confidence, unsigned long sequence) {
    return create_archived_event(frame, class_name, confidence, sequence,
                                 "road_events", "road_events.csv",
                                 config.road_snapshot_retention);
}

EventInfo create_wildlife_test_event(
        const cv::Mat& frame, const std::string& class_name,
        float confidence, unsigned long sequence) {
    return create_archived_event(
        frame, class_name, confidence, sequence,
        "wildlife_test", "wildlife_test.csv",
        config.wildlife_test_snapshot_retention);
}

bool delete_archived_event(const std::string& event_id,
                           const std::string& archive_directory,
                           const std::string& log_filename,
                           bool road_archive,
                           bool wildlife_test_archive = false) {
    if (event_id.empty() || !std::all_of(event_id.begin(), event_id.end(),
            [](unsigned char ch) { return std::isalnum(ch) || ch == '-' || ch == '_'; })) {
        return false;
    }

    std::lock_guard<std::mutex> events_lock(events_mutex);
    const auto log_path = config.output_dir / log_filename;
    std::ifstream input(log_path);
    if (!input) {
        return false;
    }

    std::vector<std::string> kept_lines;
    std::string line;
    std::string snapshot;
    bool found = false;
    while (std::getline(input, line)) {
        const std::size_t first_comma = line.find(',');
        if (first_comma != std::string::npos &&
            line.substr(0, first_comma) == event_id) {
            const std::size_t last_comma = line.rfind(',');
            if (last_comma != std::string::npos) {
                snapshot = line.substr(last_comma + 1);
            }
            found = true;
        } else {
            kept_lines.push_back(line);
        }
    }
    input.close();
    if (!found) {
        return false;
    }

    const auto temporary_path = config.output_dir / (log_filename + ".tmp");
    {
        std::ofstream output(temporary_path, std::ios::trunc);
        if (!output) {
            return false;
        }
        for (const auto& kept : kept_lines) {
            output << kept << '\n';
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary_path, log_path, error);
    if (error) {
        std::filesystem::remove(temporary_path, error);
        return false;
    }

    if (!snapshot.empty() && snapshot.find('/') == std::string::npos &&
        snapshot.find("..") == std::string::npos &&
        std::filesystem::path(snapshot).extension() == ".jpg") {
        std::filesystem::remove(config.output_dir / archive_directory / snapshot, error);
    }
    {
        std::lock_guard<std::mutex> state_lock(state.status_mutex);
        auto& recent = wildlife_test_archive
            ? state.recent_wildlife_test_events
            : (road_archive ? state.recent_road_events : state.recent_events);
        for (auto iterator = recent.begin(); iterator != recent.end();) {
            if (iterator->id == event_id) {
                iterator = recent.erase(iterator);
            } else {
                ++iterator;
            }
        }
        auto& count = wildlife_test_archive
            ? state.wildlife_test_event_count
            : (road_archive ? state.road_event_count : state.event_count);
        if (count > 0) {
            --count;
        }
    }
    return true;
}

bool delete_event(const std::string& event_id) {
    return delete_archived_event(event_id, "events", "events.csv", false);
}

bool delete_road_event(const std::string& event_id) {
    return delete_archived_event(event_id, "road_events", "road_events.csv", true);
}

bool delete_wildlife_test_event(const std::string& event_id) {
    return delete_archived_event(
        event_id, "wildlife_test", "wildlife_test.csv", false, true);
}

std::pair<std::deque<EventInfo>, unsigned long> read_event_log(
        const std::string& log_filename) {
    std::ifstream log(config.output_dir / log_filename);
    if (!log) {
        return {};
    }
    std::string line;
    std::getline(log, line);  // header
    unsigned long count = 0;
    std::deque<EventInfo> events;
    while (std::getline(log, line)) {
        std::istringstream row(line);
        EventInfo event;
        std::string confidence;
        if (!std::getline(row, event.id, ',') ||
            !std::getline(row, event.timestamp, ',') ||
            !std::getline(row, event.class_name, ',') ||
            !std::getline(row, confidence, ',') ||
            !std::getline(row, event.snapshot)) {
            continue;
        }
        try {
            event.confidence = std::stof(confidence);
        } catch (...) {
            continue;
        }
        events.push_front(std::move(event));
        while (events.size() > 20) {
            events.pop_back();
        }
        ++count;
    }
    return {std::move(events), count};
}

void load_existing_events() {
    auto driveway = read_event_log("events.csv");
    auto road = read_event_log("road_events.csv");
    auto wildlife_test = read_event_log("wildlife_test.csv");
    std::lock_guard<std::mutex> lock(state.status_mutex);
    state.recent_events = std::move(driveway.first);
    state.event_count = driveway.second;
    state.recent_road_events = std::move(road.first);
    state.road_event_count = road.second;
    state.recent_wildlife_test_events = std::move(wildlife_test.first);
    state.wildlife_test_event_count = wildlife_test.second;
}

std::string status_json() {
    bool mailbox_zone_enabled = false;
    {
        std::lock_guard<std::mutex> zone_lock(zones_mutex);
        mailbox_zone_enabled = !config.mailbox_zone.empty();
    }
    const bool storage_ready = manual_storage_ready();
    double storage_free_gb = 0.0;
    if (storage_ready) {
        std::error_code error;
        const auto space = std::filesystem::space(
            config.manual_storage_root, error);
        if (!error) {
            storage_free_gb =
                space.available / (1024.0 * 1024.0 * 1024.0);
        }
    }
    bool recording_active = false;
    double recording_elapsed = 0.0;
    std::string recording_filename;
    std::string recording_error;
    {
        std::lock_guard<std::mutex> recording_lock(manual_recording_mutex);
        recording_active = manual_recording_active;
        recording_filename = manual_recording_filename;
        recording_error = manual_recording_error;
        if (recording_active) {
            recording_elapsed = std::chrono::duration<double>(
                Clock::now() - manual_recording_started).count();
        }
    }
    std::lock_guard<std::mutex> lock(state.status_mutex);
    std::ostringstream json;
    json << std::fixed << std::setprecision(1)
         << "{\"healthy\":" << (state.camera_connected ? "true" : "false")
         << ",\"camera_fps\":" << state.camera_fps
         << ",\"inference_ms\":" << state.inference_ms
         << ",\"average_inference_ms\":" << state.average_inference_ms
         << ",\"wildlife_enabled\":"
         << (config.wildlife_model_enabled ? "true" : "false")
         << ",\"wildlife_model_ready\":"
         << (state.wildlife_model_ready ? "true" : "false")
         << ",\"wildlife_inference_ms\":" << state.wildlife_inference_ms
         << ",\"wildlife_inference_count\":" << state.wildlife_inference_count
         << ",\"species_classifier_enabled\":"
         << (config.species_classifier_enabled ? "true" : "false")
         << ",\"species_classifier_ready\":"
         << (state.species_classifier_ready ? "true" : "false")
         << ",\"species_inference_ms\":" << state.species_inference_ms
         << ",\"species_inference_count\":" << state.species_inference_count
         << ",\"last_species\":\""
         << json_escape(state.last_species) << "\""
         << ",\"last_species_confidence\":"
         << state.last_species_confidence
         << ",\"species_classifier_error\":\""
         << json_escape(state.species_classifier_error) << "\""
         << ",\"captured_frames\":" << state.captured_frames
         << ",\"inference_count\":" << state.inference_count
         << ",\"relevant_objects\":" << state.relevant_objects
         << ",\"best_person_confidence\":" << state.best_person_confidence
         << ",\"person_in_zone\":" << (state.person_in_zone ? "true" : "false")
         << ",\"mailbox_zone_enabled\":" << (mailbox_zone_enabled ? "true" : "false")
         << ",\"mailbox_vehicle_present\":" << (state.mailbox_vehicle_present ? "true" : "false")
         << ",\"mailbox_stationary_seconds\":" << state.mailbox_stationary_seconds
         << ",\"event_count\":" << state.event_count
         << ",\"road_event_count\":" << state.road_event_count
         << ",\"general_events_enabled\":"
         << (config.general_events_enabled ? "true" : "false")
         << ",\"wildlife_test_captures_enabled\":"
         << (config.wildlife_test_captures_enabled ? "true" : "false")
         << ",\"wildlife_test_event_count\":"
         << state.wildlife_test_event_count
         << ",\"manual_storage_ready\":"
         << (storage_ready ? "true" : "false")
         << ",\"manual_storage_free_gb\":" << storage_free_gb
         << ",\"manual_recording_active\":"
         << (recording_active ? "true" : "false")
         << ",\"manual_recording_elapsed\":" << recording_elapsed
         << ",\"manual_recording_seconds\":" << config.manual_recording_seconds
         << ",\"manual_recording_filename\":\""
         << json_escape(recording_filename) << "\""
         << ",\"manual_recording_error\":\""
         << json_escape(recording_error) << "\""
         << ",\"notification_enabled\":" << (ntfy_enabled() ? "true" : "false")
         << ",\"notification_count\":" << state.notification_count
         << ",\"notification_error\":\"" << json_escape(state.notification_error) << "\""
         << ",\"error\":\"" << json_escape(state.last_error) << "\"}";
    return json.str();
}

std::string archived_events_json(const std::deque<EventInfo>& events,
                                 const std::string& image_prefix,
                                 const std::string& delete_prefix) {
    std::ostringstream json;
    json << '[';
    bool first = true;
    for (const auto& event : events) {
        if (!first) {
            json << ',';
        }
        first = false;
        json << "{\"id\":\"" << json_escape(event.id)
             << "\",\"timestamp\":\"" << json_escape(event.timestamp)
             << "\",\"class_name\":\"" << json_escape(event.class_name)
             << "\",\"confidence\":" << std::fixed << std::setprecision(3)
             << event.confidence << ",\"snapshot\":\"" << image_prefix
             << json_escape(event.snapshot) << "\",\"delete_path\":\""
             << delete_prefix << json_escape(event.id) << "\"}";
    }
    json << ']';
    return json.str();
}

std::string events_json() {
    std::lock_guard<std::mutex> lock(state.status_mutex);
    return archived_events_json(state.recent_events, "/events/", "/api/events/");
}

std::string road_events_json() {
    std::lock_guard<std::mutex> lock(state.status_mutex);
    return archived_events_json(state.recent_road_events, "/road-events/",
                                "/api/road-events/");
}

std::string wildlife_test_events_json() {
    std::lock_guard<std::mutex> lock(state.status_mutex);
    return archived_events_json(
        state.recent_wildlife_test_events, "/wildlife-test/",
        "/api/wildlife-test-events/");
}

const char* dashboard_html = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driveway Watch — Pi</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#07100d;color:#edf8f3}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#17352a,#07100d 48%)}
header,main{width:min(1120px,94vw);margin:auto}header{display:flex;align-items:center;justify-content:space-between;padding:24px 0 16px}
h1,h2,p{margin:0}.live{display:flex;align-items:center;gap:8px;color:#a9f5cf}.dot{width:10px;height:10px;border-radius:50%;background:#40e58d;box-shadow:0 0 14px #40e58d}
.video{position:relative;overflow:hidden;border:1px solid #315c4b;border-radius:18px;background:#020504;box-shadow:0 20px 70px #0008}.video img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain}.boundary-canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
.capture-controls{display:flex;align-items:end;flex-wrap:wrap;gap:10px}.capture-controls .field{min-width:120px}.capture-status{min-height:1.5em;margin-top:12px;color:#a9f5cf}.manual-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:18px;padding-top:16px;border-top:1px solid #29483d}.manual-heading h3{margin:0;font-size:1rem}.manual-list{display:grid;gap:8px;margin-top:10px}.manual-capture{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border-radius:10px;background:#183126}.manual-link{min-width:0;color:#a9f5cf;text-decoration:none}.manual-link strong,.manual-link small{display:block}.manual-link strong{overflow:hidden;text-overflow:ellipsis}.manual-link small{margin-top:3px;color:#91b7a5}.capture-preview{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#07100d}.capture-actions{display:flex;gap:7px}.download{display:grid;place-items:center;flex:0 0 42px;width:42px;height:42px;border:1px solid #4d9a75;border-radius:10px;background:#24513d;color:#fff;text-decoration:none;font-size:1.1rem;cursor:pointer}.download:hover{background:#326b51}.button:disabled{cursor:not-allowed;opacity:.55}
.panel{margin:18px 0;padding:20px;border:1px solid #29483d;border-radius:16px;background:#102219dd}.panel h2{font-size:1.1rem;margin-bottom:16px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:14px;border-radius:12px;background:#183126}.label{font-size:.78rem;color:#91b7a5;text-transform:uppercase;letter-spacing:.06em}.value{font-size:1.25rem;margin-top:5px}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.tab{flex:1 1 180px;padding:13px;border:1px solid #315c4b;border-radius:12px;background:#102219;color:#a9c7ba;font-weight:700;cursor:pointer}.tab.active{background:#24513d;color:#fff;border-color:#4d9a75}[hidden]{display:none!important}.events{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.event{overflow:hidden;border-radius:12px;background:#183126}.event-image{display:block;width:100%;padding:0;border:0;background:#07100d;cursor:zoom-in}.event-image img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}.meta{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px}.meta-text{min-width:0}.event strong{display:block;text-transform:capitalize}.event small{color:#91b7a5}.delete{display:grid;place-items:center;flex:0 0 42px;width:42px;height:42px;border:1px solid #70433f;border-radius:10px;background:#3b2422;color:#ffd4cf;font-size:1.15rem;cursor:pointer}.delete:hover{background:#60302c}.note{color:#a9c7ba;line-height:1.5}.boundary-summary{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:12px}.boundary-buttons{display:flex;gap:8px}.button{min-height:44px;padding:10px 15px;border:1px solid #4d9a75;border-radius:10px;background:#24513d;color:#fff;font-weight:700;cursor:pointer}.button.secondary{border-color:#527065;background:#183126}.button.danger{border-color:#70433f;background:#3b2422;color:#ffd4cf}.editor-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field{display:grid;gap:7px;color:#a9c7ba;font-size:.85rem}.field select,.field input{width:100%;min-height:44px;padding:9px 11px;border:1px solid #527065;border-radius:10px;background:#07100d;color:#fff;font:inherit}.editor-tools,.editor-save{display:flex;flex-wrap:wrap;gap:9px;margin-top:14px}.editor-save{justify-content:flex-end;padding-top:14px;border-top:1px solid #29483d}.point-status{margin-top:12px;color:#a9f5cf}.save-status{min-height:1.5em;margin-top:10px}.save-status.error{color:#ffb3aa}.save-status.success{color:#a9f5cf}dialog.viewer{width:100vw;height:100vh;max-width:none;max-height:none;margin:0;padding:70px 18px 18px;border:0;background:#020504f2;color:#edf8f3}.viewer::backdrop{background:#000d}.viewer img{display:block;width:100%;height:calc(100vh - 110px);object-fit:contain}.viewer-bar{position:fixed;z-index:2;top:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 16px;background:#07100df2}.viewer-actions{display:flex;gap:8px}.viewer-close{display:grid;place-items:center;width:44px;height:44px;border:1px solid #527065;border-radius:10px;background:#183126;color:#fff;font-size:1.6rem;cursor:pointer}@media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}.editor-grid{grid-template-columns:1fr}.boundary-summary{align-items:flex-start;flex-direction:column}.boundary-buttons{width:100%}.boundary-buttons .button{flex:1}.capture-controls{align-items:stretch}.capture-controls .field,.capture-controls .button{flex:1 1 140px}.editor-save .button{flex:1}}
</style>
</head>
<body>
<header><h1>Driveway Watch</h1><div class="live"><span class="dot"></span><span id="health">Pi live</span></div></header>
<main>
<div class="video"><img id="live-stream" src="/stream.mjpg" alt="Live driveway camera from Raspberry Pi"><canvas class="boundary-canvas" id="boundary-canvas" hidden aria-label="Boundary drawing area"></canvas></div>
<div class="boundary-summary"><p class="note">Blue: driveway arrival. Amber: mailbox stop. Purple: passing road traffic.</p><div class="boundary-buttons"><button class="button secondary" id="toggle-boundaries" type="button">Hide boundaries</button><button class="button secondary" id="edit-boundaries" type="button">Edit boundaries</button></div></div>
<section class="panel" aria-labelledby="manual-capture-heading">
<h2 id="manual-capture-heading">Manual capture</h2>
<div class="capture-controls">
<button class="button" id="snapshot-button" type="button">Save snapshot</button>
<label class="field">Burst<select id="snapshot-burst"><option value="1">1 photo</option><option value="3">3 photos</option><option value="5">5 photos</option></select></label>
<button class="button secondary" id="record-button" type="button" aria-pressed="false">Record 30 seconds</button>
</div>
<p class="capture-status" id="capture-status" role="status">Captures save at full camera quality to the recording SSD.</p>
<p class="note" id="manual-storage">Checking SSD…</p>
</section>
<section class="panel" id="boundary-editor" hidden>
<h2>Boundary editor</h2>
<p class="note">Choose a type, press “Redraw this boundary,” then tap around the area. Nothing changes until you save.</p>
<div class="editor-grid" style="margin-top:14px">
<label class="field">Boundary type<select id="zone-type"><option value="driveway">Driveway alert</option><option value="mailbox">Mailbox stop</option><option value="road">Road archive — no alert</option></select></label>
<label class="field">Label shown on video<input id="zone-label" maxlength="48" autocomplete="off"></label>
</div>
<p class="point-status" id="point-status">Current boundary loaded</p>
<div class="editor-tools"><button class="button secondary" id="redraw-zone" type="button">Redraw this boundary</button><button class="button secondary" id="undo-point" type="button">Undo point</button></div>
<p class="save-status note" id="boundary-status" role="status"></p>
<div class="editor-save"><button class="button danger" id="restore-boundaries" type="button">Restore original boundaries</button><button class="button secondary" id="cancel-boundaries" type="button">Cancel</button><button class="button" id="save-boundaries" type="button">Save changes</button></div>
</section>
<section class="panel"><h2>Raspberry Pi NCNN preview</h2><div class="stats">
<div class="stat"><div class="label">Camera</div><div class="value" id="camera">—</div></div>
<div class="stat"><div class="label">Detection</div><div class="value" id="inference">—</div></div>
<div class="stat"><div class="label">Wildlife</div><div class="value" id="wildlife">—</div></div>
<div class="stat"><div class="label">Objects now</div><div class="value" id="objects">—</div></div>
<div class="stat"><div class="label">Frames</div><div class="value" id="frames">—</div></div>
</div></section>
<div class="tabs" role="tablist" aria-label="Event archive"><button class="tab active" id="driveway-tab" type="button" role="tab" aria-selected="true">Alerts (<span id="driveway-count">0</span>)</button><button class="tab" id="road-tab" type="button" role="tab" aria-selected="false">Road traffic (<span id="road-count">0</span>)</button><button class="tab" id="wildlife-test-tab" type="button" role="tab" aria-selected="false" hidden>Wildlife test (<span id="wildlife-test-count">0</span>)</button><button class="tab" id="captures-tab" type="button" role="tab" aria-selected="false">Captures (<span id="captures-count">0</span>)</button></div>
<div id="driveway-panel" role="tabpanel"><section class="panel"><h2>Recent alerts</h2><div class="events" id="events"><p class="note">No alerts yet</p></div></section><section class="panel"><h2>Notifications</h2><p class="note" id="notifications">Checking phone alerts…</p></section></div>
<div id="road-panel" role="tabpanel" hidden><section class="panel"><h2>Road traffic archive</h2><p class="note" style="margin-bottom:16px">Passing vehicle snapshots do not send alerts. Animals inside any boundary do.</p><div class="events" id="road-events"><p class="note">No passing vehicles captured yet</p></div></section></div>
<div id="wildlife-test-panel" role="tabpanel" hidden><section class="panel"><h2>Wildlife test candidates</h2><p class="note" style="margin-bottom:16px">Locally saved classifier tests only. Rejected candidates are labeled and never send notifications.</p><div class="events" id="wildlife-test-events"><p class="note">No wildlife candidates captured yet</p></div></section></div>
<div id="captures-panel" role="tabpanel" hidden><section class="panel"><h2>Saved captures</h2><div class="events" id="manual-captures"><p class="note">No manual captures yet</p></div></section></div>
</main>
<dialog class="viewer" id="viewer" aria-labelledby="viewer-caption">
<div class="viewer-bar"><strong id="viewer-caption">Driveway event</strong><div class="viewer-actions"><button class="delete" id="viewer-delete" type="button" aria-label="Delete this event" title="Delete this event">🗑️</button><button class="viewer-close" id="viewer-close" type="button" aria-label="Close full-screen image">×</button></div></div>
<img id="viewer-image" alt="Full-size driveway event">
</dialog>
<script>
let selectedEvent=null;
const viewer=document.getElementById('viewer');
const viewerImage=document.getElementById('viewer-image');
const viewerCaption=document.getElementById('viewer-caption');
const boundaryEditor=document.getElementById('boundary-editor');
const boundaryCanvas=document.getElementById('boundary-canvas');
const boundaryContext=boundaryCanvas.getContext('2d');
const liveStream=document.getElementById('live-stream');
const toggleBoundaries=document.getElementById('toggle-boundaries');
const zoneType=document.getElementById('zone-type');
const zoneLabel=document.getElementById('zone-label');
const pointStatus=document.getElementById('point-status');
const boundaryStatus=document.getElementById('boundary-status');
let baselineZones=[];
let workingZones=[];
let redrawing=false;
let boundariesVisible=true;
let recordingActive=false;
try{boundariesVisible=localStorage.getItem('driveway-boundaries-visible')!=='false';}catch(error){}
function refreshStreamPreference(){toggleBoundaries.textContent=boundariesVisible?'Hide boundaries':'Show boundaries';toggleBoundaries.setAttribute('aria-pressed',String(!boundariesVisible));liveStream.src='/stream.mjpg?boundaries='+(boundariesVisible?'1':'0')+'&session='+Date.now();}
async function saveSnapshotBurst(){const button=document.getElementById('snapshot-button');const status=document.getElementById('capture-status');const count=Number(document.getElementById('snapshot-burst').value)||1;button.disabled=true;status.textContent='Saving '+count+' snapshot'+(count===1?'':'s')+' to SSD…';try{const response=await fetch('/api/manual/snapshots?count='+count,{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error||'Snapshot failed.');status.textContent='Saved '+data.saved+' full-resolution snapshot'+(data.saved===1?'':'s')+' to the SSD.';await updateManualCaptures();}catch(error){status.textContent=error.message;}finally{button.disabled=false;}}
async function toggleManualRecording(){const button=document.getElementById('record-button');const status=document.getElementById('capture-status');button.disabled=true;try{const action=recordingActive?'stop':'start';const response=await fetch('/api/manual/recordings/'+action,{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error||'Recording command failed.');status.textContent=recordingActive?'Stopping and finishing the video…':'Recording camera-native video to SSD. You may close this page.';await update();}catch(error){status.textContent=error.message;}finally{button.disabled=false;}}
function formatBytes(bytes){if(bytes<1024)return bytes+' B';if(bytes<1024*1024)return (bytes/1024).toFixed(1)+' KB';if(bytes<1024*1024*1024)return (bytes/(1024*1024)).toFixed(1)+' MB';return (bytes/(1024*1024*1024)).toFixed(1)+' GB';}
async function deleteManualCapture(item){if(!confirm('Delete this '+(item.kind==='videos'?'video':'snapshot')+' from the recording SSD?'))return;const response=await fetch(item.delete_path,{method:'DELETE'});if(response.ok)await updateManualCaptures();}
async function updateManualCaptures(){const box=document.getElementById('manual-captures');try{const response=await fetch('/api/manual-captures',{cache:'no-store'});const captures=await response.json();document.getElementById('captures-count').textContent=captures.length.toLocaleString();box.replaceChildren();if(!captures.length){const empty=document.createElement('p');empty.className='note';empty.textContent='No manual captures yet';box.appendChild(empty);return;}for(const item of captures){const card=document.createElement('article');card.className='event';const preview=item.kind==='videos'?document.createElement('video'):document.createElement('img');preview.className='capture-preview';preview.src=item.url+'?preview=1';if(item.kind==='videos'){preview.controls=true;preview.preload='metadata';}else preview.alt='Saved driveway snapshot';const meta=document.createElement('div');meta.className='meta';const text=document.createElement('div');text.className='meta-text';const title=document.createElement('strong');title.textContent=item.kind==='videos'?'Video recording':'Snapshot';const details=document.createElement('small');details.textContent=new Date(item.timestamp_ms).toLocaleString()+' · '+formatBytes(item.size_bytes);text.append(title,details);const actions=document.createElement('div');actions.className='capture-actions';const download=document.createElement('a');download.className='download';download.href=item.url;download.download=item.name;download.title='Download capture';download.setAttribute('aria-label','Download capture');download.textContent='⬇️';const remove=document.createElement('button');remove.className='delete';remove.type='button';remove.title='Delete capture';remove.setAttribute('aria-label','Delete manual capture');remove.textContent='🗑️';remove.addEventListener('click',()=>deleteManualCapture(item));actions.append(download,remove);meta.append(text,actions);card.append(preview,meta);box.appendChild(card);}}catch(error){document.getElementById('captures-count').textContent='—';box.innerHTML='<p class="note">Saved captures are temporarily unavailable.</p>';}}
document.getElementById('snapshot-button').addEventListener('click',saveSnapshotBurst);
document.getElementById('record-button').addEventListener('click',toggleManualRecording);
function cloneZones(zones){return JSON.parse(JSON.stringify(zones));}
function selectedZone(){return workingZones.find(zone=>zone.id===zoneType.value);}
function resizeBoundaryCanvas(){const rectangle=liveStream.getBoundingClientRect();boundaryCanvas.width=Math.max(1,Math.round(rectangle.width));boundaryCanvas.height=Math.max(1,Math.round(rectangle.height));drawBoundaries();}
function drawBoundaries(){boundaryContext.clearRect(0,0,boundaryCanvas.width,boundaryCanvas.height);for(const zone of workingZones){if(!zone.points.length)continue;const selected=zone.id===zoneType.value;boundaryContext.beginPath();zone.points.forEach((point,index)=>{const x=point[0]*boundaryCanvas.width;const y=point[1]*boundaryCanvas.height;if(index===0)boundaryContext.moveTo(x,y);else boundaryContext.lineTo(x,y);});if(zone.points.length>=3)boundaryContext.closePath();boundaryContext.globalAlpha=selected ? .20 : .08;boundaryContext.fillStyle=zone.color;boundaryContext.fill();boundaryContext.globalAlpha=1;boundaryContext.strokeStyle=zone.color;boundaryContext.lineWidth=selected?6:4;boundaryContext.stroke();if(selected){for(const point of zone.points){boundaryContext.beginPath();boundaryContext.arc(point[0]*boundaryCanvas.width,point[1]*boundaryCanvas.height,8,0,Math.PI*2);boundaryContext.fillStyle=zone.color;boundaryContext.fill();boundaryContext.strokeStyle='#07100d';boundaryContext.lineWidth=3;boundaryContext.stroke();}}}}
function refreshZoneEditor(){const zone=selectedZone();if(!zone)return;zoneLabel.value=zone.label;redrawing=false;pointStatus.textContent=zone.points.length+' points loaded. Press “Redraw this boundary” to replace it.';boundaryStatus.className='save-status note';boundaryStatus.textContent='';drawBoundaries();}
async function openBoundaryEditor(){boundaryStatus.textContent='Loading boundaries…';try{const response=await fetch('/api/zones',{cache:'no-store'});if(!response.ok)throw new Error('load failed');const data=await response.json();baselineZones=cloneZones(data.zones);workingZones=cloneZones(data.zones);boundaryEditor.hidden=false;boundaryCanvas.hidden=false;zoneType.value='driveway';resizeBoundaryCanvas();refreshZoneEditor();boundaryEditor.scrollIntoView({behavior:'smooth',block:'nearest'});}catch(error){boundaryEditor.hidden=false;boundaryStatus.className='save-status error';boundaryStatus.textContent='The boundaries could not be loaded. No settings were changed.';}}
function closeBoundaryEditor(){workingZones=cloneZones(baselineZones);boundaryEditor.hidden=true;boundaryCanvas.hidden=true;redrawing=false;boundaryStatus.textContent='';}
function zonePointsText(points){return points.map(point=>point.map(value=>value.toFixed(5)).join(',')).join(';');}
async function saveBoundaries(){for(const zone of workingZones){if(zone.points.length<3){zoneType.value=zone.id;refreshZoneEditor();boundaryStatus.className='save-status error';boundaryStatus.textContent=zone.label+' needs at least 3 points before anything can be saved.';return;}}const form=new URLSearchParams();for(const zone of workingZones){form.set(zone.id+'_label',zone.label);form.set(zone.id+'_points',zonePointsText(zone.points));}boundaryStatus.className='save-status note';boundaryStatus.textContent='Saving…';try{const response=await fetch('/api/zones',{method:'PUT',headers:{'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'},body:form.toString()});const data=await response.json();if(!response.ok)throw new Error(data.error||'save failed');baselineZones=cloneZones(data.zones);workingZones=cloneZones(data.zones);closeBoundaryEditor();const editButton=document.getElementById('edit-boundaries');editButton.textContent='Boundaries saved ✓';setTimeout(()=>{editButton.textContent='Edit boundaries';},2500);}catch(error){boundaryStatus.className='save-status error';boundaryStatus.textContent='Nothing changed: '+error.message;}}
async function restoreBoundaries(){if(!confirm('Restore the original boundaries we calibrated? Your later boundary edits will be replaced.'))return;boundaryStatus.className='save-status note';boundaryStatus.textContent='Restoring original boundaries…';try{const response=await fetch('/api/zones/restore',{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error||'restore failed');baselineZones=cloneZones(data.zones);workingZones=cloneZones(data.zones);refreshZoneEditor();boundaryStatus.className='save-status success';boundaryStatus.textContent='Original boundaries restored and active.';}catch(error){boundaryStatus.className='save-status error';boundaryStatus.textContent='Nothing changed: '+error.message;}}
document.getElementById('edit-boundaries').addEventListener('click',openBoundaryEditor);
toggleBoundaries.addEventListener('click',()=>{boundariesVisible=!boundariesVisible;try{localStorage.setItem('driveway-boundaries-visible',String(boundariesVisible));}catch(error){}refreshStreamPreference();});
document.getElementById('cancel-boundaries').addEventListener('click',closeBoundaryEditor);
document.getElementById('save-boundaries').addEventListener('click',saveBoundaries);
document.getElementById('restore-boundaries').addEventListener('click',restoreBoundaries);
document.getElementById('redraw-zone').addEventListener('click',()=>{const zone=selectedZone();zone.points=[];redrawing=true;pointStatus.textContent='Tap at least 3 corners. Undo removes the last point.';boundaryStatus.textContent='';drawBoundaries();});
document.getElementById('undo-point').addEventListener('click',()=>{const zone=selectedZone();if(redrawing&&zone.points.length){zone.points.pop();pointStatus.textContent=zone.points.length+' point'+(zone.points.length===1?'':'s')+' placed — at least 3 required.';drawBoundaries();}});
zoneType.addEventListener('change',refreshZoneEditor);
zoneLabel.addEventListener('input',()=>{const zone=selectedZone();if(zone)zone.label=zoneLabel.value;});
boundaryCanvas.addEventListener('click',event=>{if(!redrawing)return;const rectangle=boundaryCanvas.getBoundingClientRect();const zone=selectedZone();zone.points.push([(event.clientX-rectangle.left)/rectangle.width,(event.clientY-rectangle.top)/rectangle.height]);pointStatus.textContent=zone.points.length+' point'+(zone.points.length===1?'':'s')+' placed'+(zone.points.length<3?' — at least 3 required.':' — ready to save or keep adding corners.');drawBoundaries();});
window.addEventListener('resize',()=>{if(!boundaryCanvas.hidden)resizeBoundaryCanvas();});
function eventName(event){return event.class_name.replaceAll('_',' ');}
function openEvent(event){selectedEvent=event;viewerImage.src=event.snapshot;viewerCaption.textContent=eventName(event)+' — '+Math.round(event.confidence*100)+'%';viewer.showModal();}
async function deleteEvent(event){if(!confirm('Delete this event image and its history entry? This cannot be undone.'))return;try{const response=await fetch(event.delete_path,{method:'DELETE'});if(!response.ok)throw new Error('delete failed');if(selectedEvent&&selectedEvent.id===event.id){viewer.close();selectedEvent=null;}await updateEvents();await update();}catch(error){alert('The event could not be deleted. Please try again.');}}
document.getElementById('viewer-close').addEventListener('click',()=>viewer.close());
document.getElementById('viewer-delete').addEventListener('click',()=>{if(selectedEvent)deleteEvent(selectedEvent);});
viewer.addEventListener('click',event=>{if(event.target===viewer)viewer.close();});
async function update(){try{const r=await fetch('/api/status',{cache:'no-store'});const s=await r.json();
document.getElementById('health').textContent=s.healthy?'Pi live':'Camera reconnecting';
document.getElementById('camera').textContent=s.camera_fps.toFixed(1)+' FPS';
document.getElementById('inference').textContent=s.inference_ms.toFixed(0)+' ms';
document.getElementById('wildlife').textContent=!s.wildlife_enabled?'Off':!s.wildlife_model_ready?'Detector unavailable':s.species_classifier_enabled&&!s.species_classifier_ready?'Classifier starting':s.last_species?s.last_species+' '+(s.last_species_confidence*100).toFixed(0)+'%':s.wildlife_inference_count?s.wildlife_inference_ms.toFixed(0)+' ms':'Ready';
document.getElementById('objects').textContent=s.relevant_objects;
document.getElementById('frames').textContent=s.captured_frames.toLocaleString();
const recordingWasActive=recordingActive;
recordingActive=s.manual_recording_active;
const recordButton=document.getElementById('record-button');
const captureStatus=document.getElementById('capture-status');
recordButton.textContent=recordingActive?'Stop recording':'Record 30 seconds';
recordButton.classList.toggle('danger',recordingActive);
recordButton.classList.toggle('secondary',!recordingActive);
recordButton.setAttribute('aria-pressed',String(recordingActive));
document.getElementById('manual-storage').textContent=s.manual_storage_ready?s.manual_storage_free_gb.toFixed(0)+' GB free':'SSD unavailable';
if(recordingActive)captureStatus.textContent='Recording… '+Math.min(Math.round(s.manual_recording_seconds),Math.floor(s.manual_recording_elapsed))+' of '+Math.round(s.manual_recording_seconds)+' seconds. You may close this page.';
else if(s.manual_recording_error)captureStatus.textContent=s.manual_recording_error;
else if(recordingWasActive){captureStatus.textContent='Recording saved to the SSD.';updateManualCaptures();}
document.getElementById('driveway-count').textContent=s.event_count.toLocaleString();
document.getElementById('road-count').textContent=s.road_event_count.toLocaleString();
document.getElementById('wildlife-test-count').textContent=s.wildlife_test_event_count.toLocaleString();
document.getElementById('wildlife-test-tab').hidden=!s.wildlife_test_captures_enabled;
const n=document.getElementById('notifications');
if(!s.notification_enabled)n.textContent='Phone alerts are not configured.';
else if(s.notification_error)n.textContent='Phone alerts need attention: '+s.notification_error;
else n.textContent='ntfy phone alerts are active. '+s.notification_count+' event alert'+(s.notification_count===1?' has':'s have')+' been sent since this service started.';}catch(e){document.getElementById('health').textContent='Reconnecting';}}
function renderEvents(list,box,emptyText){box.replaceChildren();if(!list.length){const p=document.createElement('p');p.className='note';p.textContent=emptyText;box.appendChild(p);return;}for(const event of list){const card=document.createElement('article');card.className='event';const imageButton=document.createElement('button');imageButton.className='event-image';imageButton.type='button';imageButton.setAttribute('aria-label','Open '+eventName(event)+' event full screen');const img=document.createElement('img');img.src=event.snapshot;img.alt=eventName(event)+' event';imageButton.appendChild(img);imageButton.addEventListener('click',()=>openEvent(event));const meta=document.createElement('div');meta.className='meta';const text=document.createElement('div');text.className='meta-text';const name=document.createElement('strong');name.textContent=eventName(event)+' — '+Math.round(event.confidence*100)+'%';const time=document.createElement('small');time.textContent=new Date(event.timestamp).toLocaleString();text.append(name,time);const actions=document.createElement('div');actions.className='capture-actions';const download=document.createElement('a');download.className='download';download.href=event.snapshot;download.download='';download.title='Download image';download.setAttribute('aria-label','Download '+eventName(event)+' image');download.textContent='⬇️';const remove=document.createElement('button');remove.className='delete';remove.type='button';remove.title='Delete this event';remove.setAttribute('aria-label','Delete '+eventName(event)+' event');remove.textContent='🗑️';remove.addEventListener('click',()=>deleteEvent(event));actions.append(download,remove);meta.append(text,actions);card.append(imageButton,meta);box.appendChild(card);}}
async function updateEvents(){try{const responses=await Promise.all([fetch('/api/events',{cache:'no-store'}),fetch('/api/road-events',{cache:'no-store'}),fetch('/api/wildlife-test-events',{cache:'no-store'})]);const lists=await Promise.all(responses.map(response=>response.json()));renderEvents(lists[0],document.getElementById('events'),'No alerts yet');renderEvents(lists[1],document.getElementById('road-events'),'No passing vehicles captured yet');renderEvents(lists[2],document.getElementById('wildlife-test-events'),'No wildlife candidates captured yet');}catch(e){}}
function selectArchive(view){const isDriveway=view==='driveway';const isRoad=view==='road';const isWildlifeTest=view==='wildlife-test';document.getElementById('driveway-panel').hidden=!isDriveway;document.getElementById('road-panel').hidden=!isRoad;document.getElementById('wildlife-test-panel').hidden=!isWildlifeTest;document.getElementById('captures-panel').hidden=view!=='captures';for(const [id,active] of [['driveway-tab',isDriveway],['road-tab',isRoad],['wildlife-test-tab',isWildlifeTest],['captures-tab',view==='captures']]){const tab=document.getElementById(id);tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));}}
document.getElementById('driveway-tab').addEventListener('click',()=>selectArchive('driveway'));document.getElementById('road-tab').addEventListener('click',()=>selectArchive('road'));document.getElementById('wildlife-test-tab').addEventListener('click',()=>selectArchive('wildlife-test'));document.getElementById('captures-tab').addEventListener('click',()=>selectArchive('captures'));
refreshStreamPreference();update();updateEvents();updateManualCaptures();setInterval(update,1000);setInterval(updateEvents,5000);setInterval(updateManualCaptures,5000);
</script>
</body></html>)HTML";

bool animal_class(int label) {
    return (label >= 14 && label <= 21) || label == 80;
}

bool relevant_class(int label) {
    return label == 0 || label == 1 || label == 2 || label == 3 ||
           label == 5 || label == 7 || animal_class(label);
}

bool mailbox_vehicle_class(int label) {
    return label == 2 || label == 5 || label == 7;
}

bool road_vehicle_class(int label) {
    return label == 1 || label == 2 || label == 3 || label == 5 || label == 7;
}

struct RoadDetection {
    const Object* object = nullptr;
    cv::Point2f center;
};

struct DetectionOverlay {
    cv::Rect box;
    std::string label;
    cv::Scalar color;
};

void draw_detection_overlay(cv::Mat& frame, const DetectionOverlay& overlay) {
    cv::rectangle(frame, overlay.box, overlay.color, 3);
    cv::putText(frame, overlay.label,
                cv::Point(overlay.box.x, std::max(24, overlay.box.y) - 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, overlay.color, 2);
}

struct RoadTrack {
    unsigned long id = 0;
    cv::Point2f center;
    Clock::time_point last_seen;
    bool matched = false;
};

const char* class_name(int label) {
    switch (label) {
        case 0: return "person";
        case 1: return "bicycle";
        case 2: return "car";
        case 3: return "motorcycle";
        case 5: return "bus";
        case 7: return "truck";
        case 14: return "bird";
        case 15: return "cat";
        case 16: return "dog";
        case 17: return "horse";
        case 18: return "sheep";
        case 19: return "cow";
        case 20: return "elephant";
        case 21: return "bear";
        case 22: return "zebra";
        case 23: return "giraffe";
        case 80: return "wildlife";
        default: return "object";
    }
}

void draw_zone_label(cv::Mat& frame, const std::string& label,
                     const cv::Point& anchor, const cv::Scalar& color) {
    if (label.empty()) {
        return;
    }

    const double resolution_scale = static_cast<double>(frame.rows) / 720.0;
    const double font_scale = std::clamp(0.5 * resolution_scale, 0.5, 1.15);
    const int thickness = std::clamp(
        static_cast<int>(std::lround(2.0 * resolution_scale)), 2, 4);
    const int padding = std::clamp(
        static_cast<int>(std::lround(5.0 * resolution_scale)), 5, 11);

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const int available_x = std::max(
        padding, frame.cols - text_size.width - padding);
    const int text_x = std::clamp(anchor.x + padding, padding, available_x);
    const int minimum_baseline = text_size.height + padding;
    const int maximum_baseline = std::max(
        minimum_baseline, frame.rows - baseline - padding);
    const int text_baseline = std::clamp(
        anchor.y - padding, minimum_baseline, maximum_baseline);

    const cv::Rect background(
        text_x - padding,
        text_baseline - text_size.height - padding,
        text_size.width + padding * 2,
        text_size.height + baseline + padding * 2);
    cv::rectangle(frame, background, cv::Scalar(8, 18, 14), cv::FILLED);
    cv::putText(frame, label, cv::Point(text_x, text_baseline),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                cv::LINE_AA);
}

void draw_zone_boundaries(cv::Mat& frame, const ZoneSettings& zones) {
    const double resolution_scale = static_cast<double>(frame.rows) / 720.0;
    const int boundary_thickness = std::clamp(
        static_cast<int>(std::lround(4.0 * resolution_scale)), 4, 9);
    std::vector<cv::Point> driveway_pixels;
    driveway_pixels.reserve(zones.driveway.size());
    for (const auto& point : zones.driveway) {
        driveway_pixels.emplace_back(
            static_cast<int>(point.x * frame.cols),
            static_cast<int>(point.y * frame.rows));
    }
    cv::polylines(frame, std::vector<std::vector<cv::Point>>{driveway_pixels}, true,
                  cv::Scalar(255, 190, 65), boundary_thickness, cv::LINE_AA);
    if (!driveway_pixels.empty()) {
        draw_zone_label(frame, zones.driveway_label, driveway_pixels.front(),
                        cv::Scalar(255, 190, 65));
    }

    if (!zones.mailbox.empty()) {
        std::vector<cv::Point> mailbox_pixels;
        mailbox_pixels.reserve(zones.mailbox.size());
        for (const auto& point : zones.mailbox) {
            mailbox_pixels.emplace_back(
                static_cast<int>(point.x * frame.cols),
                static_cast<int>(point.y * frame.rows));
        }
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{mailbox_pixels}, true,
                      cv::Scalar(0, 185, 255), boundary_thickness, cv::LINE_AA);
        if (!mailbox_pixels.empty()) {
            draw_zone_label(frame, zones.mailbox_label, mailbox_pixels.front(),
                            cv::Scalar(0, 185, 255));
        }
    }

    if (!zones.road.empty()) {
        std::vector<cv::Point> road_pixels;
        road_pixels.reserve(zones.road.size());
        for (const auto& point : zones.road) {
            road_pixels.emplace_back(
                static_cast<int>(point.x * frame.cols),
                static_cast<int>(point.y * frame.rows));
        }
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{road_pixels}, true,
                      cv::Scalar(210, 80, 210), boundary_thickness, cv::LINE_AA);
        if (!road_pixels.empty()) {
            draw_zone_label(frame, zones.road_label, road_pixels.front(),
                            cv::Scalar(210, 80, 210));
        }
    }
}

void mark_camera_disconnected(const std::string& error) {
    std::lock_guard<std::mutex> lock(state.status_mutex);
    state.camera_connected = false;
    state.camera_fps = 0.0;
    state.last_error = error;
}

void wait_for_reconnect_delay() {
    const auto delay = std::chrono::duration<double>(
        config.reconnect_delay_seconds);
    const auto deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(delay);
    while (running.load() && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void capture_loop(const std::string& camera_url) {
    while (running.load()) {
        cv::VideoCapture capture;
        const std::vector<int> options = {
            cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000,
        };
        if (!capture.open(camera_url, cv::CAP_FFMPEG, options)) {
            mark_camera_disconnected("Could not open camera; reconnecting");
            wait_for_reconnect_delay();
            continue;
        }

        unsigned long interval_frames = 0;
        auto interval_start = Clock::now();
        bool received_frame = false;
        cv::Mat frame;
        while (running.load() && capture.read(frame) && !frame.empty()) {
            if (!received_frame) {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.camera_connected = true;
                state.last_error.clear();
                received_frame = true;
            }
            {
                std::lock_guard<std::mutex> lock(state.frame_mutex);
                frame.copyTo(state.latest_frame);
                ++state.frame_generation;
            }
            ++interval_frames;
            {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                ++state.captured_frames;
            }

            const auto now = Clock::now();
            const double seconds = std::chrono::duration<double>(now - interval_start).count();
            if (seconds >= 1.0) {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.camera_fps = interval_frames / seconds;
                interval_frames = 0;
                interval_start = now;
            }
        }

        capture.release();
        if (!running.load()) {
            break;
        }
        mark_camera_disconnected("Camera stream interrupted; reconnecting");
        wait_for_reconnect_delay();
    }
}

void inference_loop() {
    YoloV8 detector;
    detector.load(640);
    WildlifeDetector wildlife_detector;
    bool wildlife_ready = false;
    if (config.wildlife_model_enabled) {
        wildlife_ready = wildlife_detector.load(
            config.wildlife_param_path, config.wildlife_model_path, 640) == 0;
        if (!wildlife_ready) {
            std::cerr << "Wildlife model unavailable; using the standard animal classes\n";
        }
    }
    {
        std::lock_guard<std::mutex> lock(state.status_mutex);
        state.wildlife_model_ready = wildlife_ready;
    }
    double total_inference_ms = 0.0;
    unsigned long event_sequence = 0;
    unsigned long road_event_sequence = 0;
    unsigned long wildlife_test_event_sequence = 0;
    unsigned long road_track_sequence = 0;
    std::vector<RoadTrack> road_tracks;
    std::map<std::string, Clock::time_point> active_classes;
    std::map<std::string, Clock::time_point> last_event_by_class;
    bool mailbox_tracking = false;
    bool mailbox_alerted_this_visit = false;
    cv::Point2f mailbox_anchor;
    auto mailbox_stationary_since = Clock::now();
    auto mailbox_last_seen = Clock::now();
    auto mailbox_last_event = Clock::now() - std::chrono::hours(24);
    float mailbox_best_confidence = 0.0f;
    const std::vector<int> jpeg_options = {cv::IMWRITE_JPEG_QUALITY, 82};
    unsigned long last_processed_generation = 0;
    std::vector<Object> cached_wildlife;
    auto next_wildlife_inference = Clock::now();
    SpeciesResult cached_species;
    cv::Rect_<float> cached_species_rect;
    auto cached_species_until = Clock::now();
    auto next_species_attempt = Clock::now();
    auto last_wildlife_test_capture =
        Clock::now() - std::chrono::hours(24);

    struct PresentDetection {
        const Object* object = nullptr;
        float event_confidence = 0.0f;
        std::string event_name;
    };

    while (running.load()) {
        const auto cycle_start = Clock::now();
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(state.frame_mutex);
            if (!state.latest_frame.empty() &&
                state.frame_generation != last_processed_generation) {
                state.latest_frame.copyTo(frame);
                last_processed_generation = state.frame_generation;
            }
        }
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            continue;
        }

        ZoneSettings zones;
        {
            std::lock_guard<std::mutex> lock(zones_mutex);
            zones = current_zones_locked();
        }

        std::vector<Object> objects;
        const auto inference_start = Clock::now();
        detector.detect(frame, objects,
                        std::min({config.confidence_threshold,
                                  config.person_confidence_threshold,
                                  config.animal_confidence_threshold}),
                        config.nms_threshold);
        const double inference_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - inference_start).count();
        if (wildlife_ready) {
            objects.erase(
                std::remove_if(objects.begin(), objects.end(),
                               [](const Object& object) {
                                   return animal_class(object.label);
                               }),
                objects.end());
            const auto wildlife_now = Clock::now();
            if (wildlife_now >= next_wildlife_inference) {
                const auto wildlife_started = Clock::now();
                std::vector<Object> detected_wildlife;
                if (wildlife_detector.detect(
                        frame, detected_wildlife,
                        config.wildlife_confidence_threshold,
                        config.nms_threshold) == 0) {
                    cached_wildlife = std::move(detected_wildlife);
                }
                const double wildlife_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - wildlife_started).count();
                {
                    std::lock_guard<std::mutex> lock(state.status_mutex);
                    state.wildlife_inference_ms = wildlife_ms;
                    ++state.wildlife_inference_count;
                }
                const auto wildlife_interval = std::chrono::duration<double>(
                    1.0 / config.wildlife_detection_fps);
                next_wildlife_inference =
                    wildlife_now +
                    std::chrono::duration_cast<Clock::duration>(
                        wildlife_interval);
            }
            objects.insert(
                objects.end(), cached_wildlife.begin(), cached_wildlife.end());
        }

        int relevant_count = 0;
        double best_person_confidence = 0.0;
        bool person_in_zone = false;
        const Object* mailbox_vehicle = nullptr;
        cv::Point2f mailbox_vehicle_center;
        std::vector<RoadDetection> road_detections;
        std::vector<DetectionOverlay> detection_overlays;
        std::map<std::string, PresentDetection> present_classes;
        for (const auto& object : objects) {
            if (!relevant_class(object.label)) {
                continue;
            }
            if (object.label == 0) {
                best_person_confidence = std::max(
                    best_person_confidence, static_cast<double>(object.prob));
            }
            const float required_confidence = object.label == 0
                ? config.person_confidence_threshold
                : (animal_class(object.label)
                       ? (object.label == 80
                              ? config.wildlife_confidence_threshold
                              : config.animal_confidence_threshold)
                       : config.confidence_threshold);
            if (object.prob < required_confidence) {
                continue;
            }
            const cv::Point2f zone_point(
                (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                (object.rect.y + object.rect.height) / frame.rows);
            const bool inside_driveway =
                cv::pointPolygonTest(zones.driveway, zone_point, false) >= 0;
            const bool inside_mailbox = !zones.mailbox.empty() &&
                cv::pointPolygonTest(zones.mailbox, zone_point, false) >= 0;
            const bool inside_road = !zones.road.empty() &&
                cv::pointPolygonTest(zones.road, zone_point, false) >= 0;

            if (animal_class(object.label)) {
                if (!inside_driveway && !inside_mailbox && !inside_road) {
                    continue;
                }
                SpeciesResult species;
                if (object.label == 80 && config.species_classifier_enabled) {
                    const cv::Rect_<float> intersection =
                        object.rect & cached_species_rect;
                    const float union_area = object.rect.area() +
                        cached_species_rect.area() - intersection.area();
                    const float overlap = union_area > 0.0f
                        ? intersection.area() / union_area : 0.0f;
                    const auto species_now = Clock::now();
                    if (cached_species.available &&
                        species_now < cached_species_until &&
                        overlap >= 0.25f) {
                        species = cached_species;
                    } else if (species_now >= next_species_attempt) {
                        const float padding_x = object.rect.width * 0.12f;
                        const float padding_y = object.rect.height * 0.12f;
                        const int crop_x = std::max(
                            0, static_cast<int>(std::floor(
                                   object.rect.x - padding_x)));
                        const int crop_y = std::max(
                            0, static_cast<int>(std::floor(
                                   object.rect.y - padding_y)));
                        const int crop_right = std::min(
                            frame.cols, static_cast<int>(std::ceil(
                                object.rect.x + object.rect.width + padding_x)));
                        const int crop_bottom = std::min(
                            frame.rows, static_cast<int>(std::ceil(
                                object.rect.y + object.rect.height + padding_y)));
                        const auto species_started = Clock::now();
                        if (crop_right > crop_x && crop_bottom > crop_y) {
                            species = classify_species_crop(frame(
                                cv::Rect(crop_x, crop_y,
                                         crop_right - crop_x,
                                         crop_bottom - crop_y)));
                        } else {
                            species.error = "invalid animal crop";
                        }
                        const double species_ms =
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - species_started).count();
                        {
                            std::lock_guard<std::mutex> lock(
                                state.status_mutex);
                            state.species_classifier_ready =
                                species.available;
                            state.species_inference_ms = species_ms;
                            ++state.species_inference_count;
                            state.species_classifier_error = species.error;
                            if (species.available) {
                                state.last_species = species.label;
                                state.last_species_confidence =
                                    species.confidence;
                            }
                        }
                        next_species_attempt = species_now +
                            std::chrono::seconds(
                                species.available ? 1 : 5);
                        if (species.available) {
                            cached_species = species;
                            cached_species_rect = object.rect;
                            cached_species_until = species_now +
                                std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(
                                        config.species_cache_seconds));
                        }
                    }
                }

                const bool species_accepted =
                    !config.species_classifier_enabled ||
                    (species.available &&
                     species.confidence >=
                         config.species_confidence_threshold &&
                     !rejected_species_label(species.label));
                const bool confirmed = species_accepted ||
                    !config.species_classifier_required;
                if (confirmed) {
                    ++relevant_count;
                }
                std::string event_name = "animal_" +
                    std::string(class_name(object.label));
                float event_confidence = object.prob;
                if (species_accepted &&
                    config.species_classifier_enabled) {
                    event_name = "animal_" + species_slug(species.label);
                    event_confidence = species.confidence;
                }
                if (confirmed && config.animal_alerts_enabled) {
                    const auto existing = present_classes.find(event_name);
                    if (existing == present_classes.end() ||
                        existing->second.event_confidence <
                            event_confidence) {
                        present_classes[event_name] = {
                            &object, event_confidence, event_name};
                    }
                }
                const cv::Scalar color = !confirmed
                    ? cv::Scalar(130, 130, 130)
                    : (inside_driveway
                    ? cv::Scalar(65, 229, 141)
                    : (inside_mailbox ? cv::Scalar(0, 185, 255)
                                      : cv::Scalar(210, 80, 210)));
                std::ostringstream label;
                if (species.available &&
                    config.species_classifier_enabled) {
                    label << species.label << " " << std::fixed
                          << std::setprecision(0)
                          << species.confidence * 100.0f << "% / detector ";
                } else if (config.species_classifier_enabled) {
                    label << "wildlife candidate ";
                } else if (object.label == 80) {
                    label << "wildlife ";
                } else {
                    label << "animal " << class_name(object.label) << " ";
                }
                label << std::fixed
                      << std::setprecision(0) << object.prob * 100.0f << "%";
                if (!confirmed) {
                    label << " rejected";
                } else if (!config.animal_alerts_enabled) {
                    label << " test only";
                }
                const std::string overlay_label = label.str();
                detection_overlays.push_back(
                    {object.rect, overlay_label, color});
                const double since_wildlife_test =
                    std::chrono::duration<double>(
                        Clock::now() - last_wildlife_test_capture).count();
                if (config.wildlife_test_captures_enabled &&
                    species.available &&
                    since_wildlife_test >=
                        config.wildlife_test_cooldown_seconds) {
                    cv::Mat test_snapshot;
                    frame.copyTo(test_snapshot);
                    draw_detection_overlay(
                        test_snapshot,
                        {object.rect, overlay_label, color});
                    std::string test_class = species_slug(species.label);
                    if (!species_accepted) {
                        test_class += "_rejected";
                    }
                    EventInfo test_event = create_wildlife_test_event(
                        test_snapshot, test_class, species.confidence,
                        ++wildlife_test_event_sequence);
                    {
                        std::lock_guard<std::mutex> lock(state.status_mutex);
                        state.recent_wildlife_test_events.push_front(
                            std::move(test_event));
                        while (state.recent_wildlife_test_events.size() > 20) {
                            state.recent_wildlife_test_events.pop_back();
                        }
                        ++state.wildlife_test_event_count;
                    }
                    last_wildlife_test_capture = Clock::now();
                }
                continue;
            }

            const bool in_mailbox = mailbox_vehicle_class(object.label) && inside_mailbox;
            if (in_mailbox && (!mailbox_vehicle || mailbox_vehicle->prob < object.prob)) {
                mailbox_vehicle = &object;
                mailbox_vehicle_center = cv::Point2f(
                    (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                    (object.rect.y + object.rect.height * 0.5f) / frame.rows);
            }
            const bool in_road = road_vehicle_class(object.label) && inside_road;
            if (in_road) {
                road_detections.push_back({&object, cv::Point2f(
                    (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                    (object.rect.y + object.rect.height * 0.5f) / frame.rows)});
            }
            if (!inside_driveway) {
                if (in_mailbox || in_road) {
                    ++relevant_count;
                    const cv::Scalar color = in_mailbox
                        ? cv::Scalar(0, 185, 255) : cv::Scalar(210, 80, 210);
                    std::ostringstream label;
                    label << (in_mailbox ? "mailbox " : "road ")
                          << class_name(object.label) << " " << std::fixed
                          << std::setprecision(0) << object.prob * 100.0f << "%";
                    detection_overlays.push_back({object.rect, label.str(), color});
                }
                continue;
            }
            if (object.label == 0) {
                person_in_zone = true;
            }
            ++relevant_count;
            const std::string name = class_name(object.label);
            if (config.general_events_enabled) {
                const auto existing = present_classes.find(name);
                if (existing == present_classes.end() ||
                    existing->second.event_confidence < object.prob) {
                    present_classes[name] = {&object, object.prob, name};
                }
            }
            std::ostringstream label;
            label << name << " " << std::fixed
                  << std::setprecision(0) << object.prob * 100.0f << "%";
            detection_overlays.push_back(
                {object.rect, label.str(), cv::Scalar(65, 229, 141)});
        }

        const auto event_now = Clock::now();
        for (auto& track : road_tracks) {
            track.matched = false;
        }
        road_tracks.erase(std::remove_if(road_tracks.begin(), road_tracks.end(),
            [&](const RoadTrack& track) {
                return std::chrono::duration<double>(event_now - track.last_seen).count() >=
                    config.road_track_lost_seconds;
            }), road_tracks.end());
        for (const auto& detection : road_detections) {
            RoadTrack* closest = nullptr;
            double closest_distance = config.road_track_match_distance;
            for (auto& track : road_tracks) {
                if (track.matched) {
                    continue;
                }
                const double distance = cv::norm(detection.center - track.center);
                if (distance < closest_distance) {
                    closest = &track;
                    closest_distance = distance;
                }
            }
            if (closest) {
                closest->center = detection.center;
                closest->last_seen = event_now;
                closest->matched = true;
                continue;
            }

            road_tracks.push_back({++road_track_sequence, detection.center, event_now, true});
            if (!config.general_events_enabled) {
                continue;
            }
            const cv::Scalar road_color(210, 80, 210);
            std::ostringstream road_label;
            road_label << "road " << class_name(detection.object->label) << " "
                       << std::fixed << std::setprecision(0)
                       << detection.object->prob * 100.0f << "%";
            cv::Mat road_snapshot;
            frame.copyTo(road_snapshot);
            draw_detection_overlay(
                road_snapshot,
                {detection.object->rect, road_label.str(), road_color});
            EventInfo road_event = create_road_event(
                road_snapshot, class_name(detection.object->label),
                detection.object->prob, ++road_event_sequence);
            {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.recent_road_events.push_front(std::move(road_event));
                while (state.recent_road_events.size() > 20) {
                    state.recent_road_events.pop_back();
                }
                ++state.road_event_count;
            }
        }
        for (const auto& overlay : detection_overlays) {
            draw_detection_overlay(frame, overlay);
        }
        double mailbox_stationary_seconds = 0.0;
        if (mailbox_vehicle) {
            if (!mailbox_tracking) {
                mailbox_tracking = true;
                mailbox_alerted_this_visit = false;
                mailbox_anchor = mailbox_vehicle_center;
                mailbox_stationary_since = event_now;
                mailbox_best_confidence = mailbox_vehicle->prob;
            } else {
                const double movement = cv::norm(mailbox_vehicle_center - mailbox_anchor);
                if (movement > config.mailbox_max_movement) {
                    mailbox_anchor = mailbox_vehicle_center;
                    mailbox_stationary_since = event_now;
                    mailbox_best_confidence = mailbox_vehicle->prob;
                } else {
                    mailbox_best_confidence = std::max(
                        mailbox_best_confidence, mailbox_vehicle->prob);
                }
            }
            mailbox_last_seen = event_now;
            mailbox_stationary_seconds =
                std::chrono::duration<double>(event_now - mailbox_stationary_since).count();
            const double since_mailbox_event =
                std::chrono::duration<double>(event_now - mailbox_last_event).count();
            if (config.general_events_enabled &&
                !mailbox_alerted_this_visit &&
                mailbox_stationary_seconds >= config.mailbox_dwell_seconds &&
                since_mailbox_event >= config.mailbox_cooldown_seconds) {
                mailbox_alerted_this_visit = true;
                mailbox_last_event = event_now;
                EventInfo event = create_event(
                    frame, "mailbox_vehicle", mailbox_best_confidence, ++event_sequence);
                queue_notification(event);
                {
                    std::lock_guard<std::mutex> lock(state.status_mutex);
                    state.recent_events.push_front(std::move(event));
                    while (state.recent_events.size() > 20) {
                        state.recent_events.pop_back();
                    }
                    ++state.event_count;
                }
            }
        } else if (mailbox_tracking &&
                   std::chrono::duration<double>(event_now - mailbox_last_seen).count() >=
                       config.mailbox_lost_seconds) {
            mailbox_tracking = false;
            mailbox_alerted_this_visit = false;
            mailbox_best_confidence = 0.0f;
        }
        for (auto iterator = active_classes.begin(); iterator != active_classes.end();) {
            const double missing_seconds =
                std::chrono::duration<double>(event_now - iterator->second).count();
            if (present_classes.find(iterator->first) == present_classes.end() &&
                missing_seconds >= config.event_clear_seconds) {
                iterator = active_classes.erase(iterator);
            } else {
                ++iterator;
            }
        }
        for (const auto& [name, present] : present_classes) {
            const bool was_active = active_classes.find(name) != active_classes.end();
            active_classes[name] = event_now;
            const auto previous_event = last_event_by_class.find(name);
            const double since_last_event = previous_event == last_event_by_class.end()
                ? 1e12
                : std::chrono::duration<double>(event_now - previous_event->second).count();
            if (was_active || since_last_event < config.event_cooldown_seconds) {
                continue;
            }
            last_event_by_class[name] = event_now;
            EventInfo event = create_event(
                frame, present.event_name, present.event_confidence,
                ++event_sequence);
            queue_notification(event);
            {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.recent_events.push_front(std::move(event));
                while (state.recent_events.size() > 20) {
                    state.recent_events.pop_back();
                }
                ++state.event_count;
            }
        }

        std::vector<unsigned char> clean_jpeg;
        if (clean_stream_clients.load() > 0) {
            cv::imencode(".jpg", frame, clean_jpeg, jpeg_options);
        }
        cv::Mat annotated_frame;
        frame.copyTo(annotated_frame);
        draw_zone_boundaries(annotated_frame, zones);
        std::vector<unsigned char> jpeg;
        cv::imencode(".jpg", annotated_frame, jpeg, jpeg_options);
        {
            std::lock_guard<std::mutex> lock(state.frame_mutex);
            state.latest_jpeg = std::move(jpeg);
            state.latest_clean_jpeg = std::move(clean_jpeg);
            ++state.jpeg_generation;
        }
        {
            std::lock_guard<std::mutex> lock(state.status_mutex);
            ++state.inference_count;
            total_inference_ms += inference_ms;
            state.inference_ms = inference_ms;
            state.average_inference_ms = total_inference_ms / state.inference_count;
            state.relevant_objects = relevant_count;
            state.best_person_confidence = best_person_confidence;
            state.person_in_zone = person_in_zone;
            state.mailbox_vehicle_present = mailbox_vehicle != nullptr;
            state.mailbox_stationary_seconds = mailbox_stationary_seconds;
        }

        const auto detection_interval = std::chrono::duration<double>(
            1.0 / config.detection_fps);
        const auto next_cycle = cycle_start +
            std::chrono::duration_cast<Clock::duration>(detection_interval);
        std::this_thread::sleep_until(next_cycle);
    }
}

void serve_mjpeg(int client_socket, bool show_boundaries) {
    const bool clean_stream = !show_boundaries;
    if (clean_stream) {
        clean_stream_clients.fetch_add(1);
    }
    const std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n";
    if (!send_all(client_socket, headers.data(), headers.size())) {
        if (clean_stream) {
            clean_stream_clients.fetch_sub(1);
        }
        return;
    }

    unsigned long last_generation = 0;
    while (running.load()) {
        std::vector<unsigned char> jpeg;
        unsigned long generation = 0;
        {
            std::lock_guard<std::mutex> lock(state.frame_mutex);
            generation = state.jpeg_generation;
            if (generation != last_generation) {
                jpeg = show_boundaries ? state.latest_jpeg : state.latest_clean_jpeg;
            }
        }
        if (jpeg.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        std::ostringstream part_headers;
        part_headers << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                     << jpeg.size() << "\r\n\r\n";
        const std::string part = part_headers.str();
        if (!send_all(client_socket, part.data(), part.size()) ||
            !send_all(client_socket, jpeg.data(), jpeg.size()) ||
            !send_all(client_socket, "\r\n", 2)) {
            break;
        }
        last_generation = generation;
    }
    if (clean_stream) {
        clean_stream_clients.fetch_sub(1);
    }
}

void serve_archived_image(int client_socket, const std::string& archive_directory,
                          const std::string& filename) {
    if (filename.empty() || filename.find('/') != std::string::npos ||
        filename.find("..") != std::string::npos ||
        std::filesystem::path(filename).extension() != ".jpg") {
        send_text(client_socket, "text/plain", "Not found\n", "404 Not Found");
        return;
    }
    std::ifstream image(config.output_dir / archive_directory / filename,
                        std::ios::binary);
    if (!image) {
        send_text(client_socket, "text/plain", "Not found\n", "404 Not Found");
        return;
    }
    const std::string body((std::istreambuf_iterator<char>(image)),
                           std::istreambuf_iterator<char>());
    send_text(client_socket, "image/jpeg", body);
}

void serve_manual_capture(int client_socket, const std::string& kind,
                          const std::string& filename, bool download) {
    if (!manual_storage_ready() || !valid_manual_capture(kind, filename)) {
        send_text(client_socket, "text/plain", "Not found\n", "404 Not Found");
        return;
    }
    const auto path = manual_directory(kind) / filename;
    std::ifstream file(path, std::ios::binary);
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (!file || error) {
        send_text(client_socket, "text/plain", "Not found\n", "404 Not Found");
        return;
    }
    const std::string content_type =
        kind == "snapshots" ? "image/jpeg" : "video/mp4";
    std::ostringstream headers;
    headers << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << size << "\r\n"
            << "Content-Disposition: "
            << (download ? "attachment" : "inline") << "; filename=\""
            << filename << "\"\r\n"
            << "Cache-Control: private, no-store\r\n"
            << "Connection: close\r\n\r\n";
    const std::string header_text = headers.str();
    if (!send_all(client_socket, header_text.data(), header_text.size())) {
        return;
    }
    std::array<char, 65536> buffer{};
    while (file) {
        file.read(buffer.data(), buffer.size());
        const auto bytes = file.gcount();
        if (bytes <= 0 ||
            !send_all(client_socket, buffer.data(),
                      static_cast<std::size_t>(bytes))) {
            break;
        }
    }
}

void handle_client(int client_socket) {
    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    char request_buffer[8192]{};
    const ssize_t received = recv(client_socket, request_buffer,
                                  sizeof(request_buffer), 0);
    if (received <= 0) {
        close(client_socket);
        return;
    }

    std::string request_text(request_buffer, static_cast<std::size_t>(received));
    const std::size_t header_end = request_text.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        send_text(client_socket, "text/plain", "Bad request\n", "400 Bad Request");
        close(client_socket);
        return;
    }
    std::size_t content_length = 0;
    std::string lowercase_headers = request_text.substr(0, header_end);
    std::transform(lowercase_headers.begin(), lowercase_headers.end(),
                   lowercase_headers.begin(), [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    const std::string length_header = "\r\ncontent-length:";
    const std::size_t length_position = lowercase_headers.find(length_header);
    if (length_position != std::string::npos) {
        const std::size_t number_start = length_position + length_header.size();
        const std::size_t number_end = lowercase_headers.find("\r\n", number_start);
        try {
            content_length = static_cast<std::size_t>(std::stoul(
                lowercase_headers.substr(number_start, number_end - number_start)));
        } catch (...) {
            content_length = 0;
        }
    }
    if (content_length > 65536) {
        send_text(client_socket, "text/plain", "Request too large\n",
                  "413 Content Too Large");
        close(client_socket);
        return;
    }
    const std::size_t body_start = header_end + 4;
    while (request_text.size() < body_start + content_length) {
        const ssize_t more = recv(client_socket, request_buffer,
                                  sizeof(request_buffer), 0);
        if (more <= 0) {
            break;
        }
        request_text.append(request_buffer, static_cast<std::size_t>(more));
    }
    const std::string request_body = request_text.size() >= body_start
        ? request_text.substr(body_start, content_length) : std::string{};

    std::istringstream request(request_text.substr(0, header_end));
    std::string method;
    std::string path;
    request >> method >> path;
    std::string query;
    const std::size_t query_start = path.find('?');
    if (query_start != std::string::npos) {
        query = path.substr(query_start + 1);
        path = path.substr(0, query_start);
    }

    if (method == "PUT" && path == "/api/zones") {
        std::string error_message;
        if (update_zones_from_form(request_body, error_message)) {
            send_text(client_socket, "application/json", zones_json());
        } else {
            send_text(client_socket, "application/json",
                      "{\"error\":\"" + json_escape(error_message) + "\"}",
                      "400 Bad Request");
        }
    } else if (method == "POST" && path == "/api/zones/restore") {
        if (restore_default_zones()) {
            send_text(client_socket, "application/json", zones_json());
        } else {
            send_text(client_socket, "application/json",
                      "{\"error\":\"The original boundaries could not be restored.\"}",
                      "500 Internal Server Error");
        }
    } else if (method == "POST" && path == "/api/manual/snapshots") {
        int count = 1;
        const auto values = parse_form(query);
        const auto count_value = values.find("count");
        if (count_value != values.end()) {
            try {
                count = std::stoi(count_value->second);
            } catch (...) {
                count = 1;
            }
        }
        std::string error_message;
        const int saved = save_manual_snapshots(count, error_message);
        if (saved > 0) {
            send_text(client_socket, "application/json",
                      "{\"saved\":" + std::to_string(saved) + "}");
        } else {
            send_text(client_socket, "application/json",
                      "{\"error\":\"" + json_escape(error_message) + "\"}",
                      "503 Service Unavailable");
        }
    } else if (method == "POST" &&
               path == "/api/manual/recordings/start") {
        std::string error_message;
        if (start_manual_recording(error_message)) {
            send_text(client_socket, "application/json", "{\"started\":true}");
        } else {
            send_text(client_socket, "application/json",
                      "{\"error\":\"" + json_escape(error_message) + "\"}",
                      "409 Conflict");
        }
    } else if (method == "POST" &&
               path == "/api/manual/recordings/stop") {
        std::string error_message;
        if (stop_manual_recording(error_message)) {
            send_text(client_socket, "application/json", "{\"stopping\":true}");
        } else {
            send_text(client_socket, "application/json",
                      "{\"error\":\"" + json_escape(error_message) + "\"}",
                      "409 Conflict");
        }
    } else if (method == "DELETE" &&
               path.rfind("/api/manual-captures/", 0) == 0) {
        const std::string remainder =
            path.substr(std::strlen("/api/manual-captures/"));
        const std::size_t separator = remainder.find('/');
        const std::string kind = remainder.substr(0, separator);
        const std::string filename = separator == std::string::npos
            ? std::string{} : remainder.substr(separator + 1);
        if (delete_manual_capture(kind, filename)) {
            send_text(client_socket, "application/json", "{\"deleted\":true}");
        } else {
            send_text(client_socket, "application/json", "{\"deleted\":false}",
                      "404 Not Found");
        }
    } else if (method == "DELETE" && path.rfind("/api/road-events/", 0) == 0) {
        const std::string event_id = path.substr(std::strlen("/api/road-events/"));
        if (delete_road_event(event_id)) {
            send_text(client_socket, "application/json", "{\"deleted\":true}");
        } else {
            send_text(client_socket, "application/json", "{\"deleted\":false}",
                      "404 Not Found");
        }
    } else if (method == "DELETE" &&
               path.rfind("/api/wildlife-test-events/", 0) == 0) {
        const std::string event_id =
            path.substr(std::strlen("/api/wildlife-test-events/"));
        if (delete_wildlife_test_event(event_id)) {
            send_text(client_socket, "application/json", "{\"deleted\":true}");
        } else {
            send_text(client_socket, "application/json", "{\"deleted\":false}",
                      "404 Not Found");
        }
    } else if (method == "DELETE" && path.rfind("/api/events/", 0) == 0) {
        const std::string event_id = path.substr(std::strlen("/api/events/"));
        if (delete_event(event_id)) {
            send_text(client_socket, "application/json", "{\"deleted\":true}");
        } else {
            send_text(client_socket, "application/json", "{\"deleted\":false}",
                      "404 Not Found");
        }
    } else if (method != "GET") {
        send_text(client_socket, "text/plain", "Method not allowed\n",
                  "405 Method Not Allowed");
    } else if (path == "/" || path == "/index.html") {
        send_text(client_socket, "text/html; charset=utf-8", dashboard_html);
    } else if (path == "/api/status") {
        send_text(client_socket, "application/json", status_json());
    } else if (path == "/api/events") {
        send_text(client_socket, "application/json", events_json());
    } else if (path == "/api/road-events") {
        send_text(client_socket, "application/json", road_events_json());
    } else if (path == "/api/wildlife-test-events") {
        send_text(
            client_socket, "application/json", wildlife_test_events_json());
    } else if (path == "/api/manual-captures") {
        send_text(client_socket, "application/json", manual_captures_json());
    } else if (path == "/api/zones") {
        send_text(client_socket, "application/json", zones_json());
    } else if (path == "/stream.mjpg") {
        serve_mjpeg(client_socket, query.find("boundaries=0") == std::string::npos);
    } else if (path.rfind("/events/", 0) == 0) {
        const std::string filename = path.substr(std::strlen("/events/"));
        serve_archived_image(client_socket, "events", filename);
    } else if (path.rfind("/road-events/", 0) == 0) {
        const std::string filename = path.substr(std::strlen("/road-events/"));
        serve_archived_image(client_socket, "road_events", filename);
    } else if (path.rfind("/wildlife-test/", 0) == 0) {
        const std::string filename =
            path.substr(std::strlen("/wildlife-test/"));
        serve_archived_image(client_socket, "wildlife_test", filename);
    } else if (path.rfind("/manual/", 0) == 0) {
        const std::string remainder =
            path.substr(std::strlen("/manual/"));
        const std::size_t separator = remainder.find('/');
        const std::string kind = remainder.substr(0, separator);
        const std::string filename = separator == std::string::npos
            ? std::string{} : remainder.substr(separator + 1);
        serve_manual_capture(
            client_socket, kind, filename,
            query.find("preview=1") == std::string::npos);
    } else if (path == "/favicon.ico") {
        send_text(client_socket, "text/plain", "", "204 No Content");
    } else {
        send_text(client_socket, "text/plain", "Not found\n", "404 Not Found");
    }
    close(client_socket);
}

int web_loop(int port) {
    const int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Could not create web socket\n";
        return 1;
    }
    int reuse = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(server_socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(server_socket, 8) < 0) {
        std::cerr << "Could not listen on port " << port << "\n";
        close(server_socket);
        return 1;
    }

    std::cout << "Driveway Pi preview listening on port " << port << std::endl;
    while (running.load()) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_socket, &read_fds);
        timeval wait_time{};
        wait_time.tv_sec = 1;
        const int ready = select(server_socket + 1, &read_fds, nullptr, nullptr, &wait_time);
        if (ready <= 0) {
            continue;
        }
        const int client_socket = accept(server_socket, nullptr, nullptr);
        if (client_socket >= 0) {
            std::thread(handle_client, client_socket).detach();
        }
    }
    close(server_socket);
    return 0;
}
}

int main(int argc, char** argv) {
    const char* camera_env = std::getenv("CAMERA_URL");
    const std::string camera_url = camera_env ? camera_env : "";
    config.camera_url = camera_url;
    config.driveway_zone = parse_zone(std::getenv("DRIVEWAY_ZONE"));
    config.mailbox_zone = parse_optional_zone(std::getenv("MAILBOX_ZONE"));
    config.road_zone = parse_optional_zone(std::getenv("ROAD_ZONE"));
    config.road_track_lost_seconds = std::max(
        0.2, environment_number("ROAD_TRACK_LOST_SECONDS", 0.8));
    config.road_track_match_distance = std::clamp(
        environment_number("ROAD_TRACK_MATCH_DISTANCE", 0.20), 0.03, 0.50);
    config.road_snapshot_retention = static_cast<std::size_t>(std::max(
        100.0, environment_number("ROAD_SNAPSHOT_RETENTION", 100000.0)));
    config.mailbox_dwell_seconds = std::max(
        0.5, environment_number("MAILBOX_DWELL_SECONDS", 2.5));
    config.mailbox_lost_seconds = std::max(
        0.2, environment_number("MAILBOX_LOST_SECONDS", 1.0));
    config.mailbox_cooldown_seconds = std::max(
        0.0, environment_number("MAILBOX_COOLDOWN_SECONDS", 300.0));
    config.mailbox_max_movement = std::clamp(
        environment_number("MAILBOX_MAX_MOVEMENT", 0.035), 0.005, 0.25);
    config.event_cooldown_seconds = std::max(
        0.0, environment_number("EVENT_COOLDOWN_SECONDS", 60.0));
    config.event_clear_seconds = std::max(
        0.0, environment_number("EVENT_CLEAR_SECONDS", 5.0));
    config.confidence_threshold = static_cast<float>(std::clamp(
        environment_number("CONFIDENCE_THRESHOLD", 0.40), 0.01, 0.99));
    config.person_confidence_threshold = static_cast<float>(std::clamp(
        environment_number("PERSON_CONFIDENCE_THRESHOLD", 0.25), 0.01, 0.99));
    config.animal_confidence_threshold = static_cast<float>(std::clamp(
        environment_number("ANIMAL_CONFIDENCE_THRESHOLD", 0.35), 0.01, 0.99));
    config.animal_alerts_enabled =
        environment_flag("ANIMAL_ALERTS_ENABLED", true);
    config.general_events_enabled =
        environment_flag("GENERAL_EVENTS_ENABLED", true);
    config.wildlife_confidence_threshold = static_cast<float>(std::clamp(
        environment_number("WILDLIFE_CONFIDENCE_THRESHOLD", 0.20), 0.01, 0.99));
    config.nms_threshold = static_cast<float>(std::clamp(
        environment_number("NMS_THRESHOLD", 0.50), 0.01, 0.99));
    config.detection_fps = std::clamp(
        environment_number("DETECTION_FPS", 5.0), 1.0, 15.0);
    config.wildlife_model_enabled =
        environment_flag("WILDLIFE_MODEL_ENABLED", false);
    config.wildlife_detection_fps = std::clamp(
        environment_number("WILDLIFE_DETECTION_FPS", 1.0), 0.2, 5.0);
    const char* wildlife_param_env = std::getenv("WILDLIFE_PARAM_PATH");
    const char* wildlife_model_env = std::getenv("WILDLIFE_MODEL_PATH");
    if (wildlife_param_env && *wildlife_param_env) {
        config.wildlife_param_path = wildlife_param_env;
    }
    if (wildlife_model_env && *wildlife_model_env) {
        config.wildlife_model_path = wildlife_model_env;
    }
    config.species_classifier_enabled =
        environment_flag("SPECIES_CLASSIFIER_ENABLED", false);
    config.species_classifier_required =
        environment_flag("SPECIES_CLASSIFIER_REQUIRED", true);
    config.species_classifier_port = static_cast<int>(std::clamp(
        environment_number("SPECIES_CLASSIFIER_PORT", 8765.0),
        1024.0, 65535.0));
    config.species_confidence_threshold = static_cast<float>(std::clamp(
        environment_number("SPECIES_CONFIDENCE_THRESHOLD", 0.65),
        0.01, 0.99));
    config.species_classifier_timeout_seconds = std::clamp(
        environment_number("SPECIES_CLASSIFIER_TIMEOUT_SECONDS", 8.0),
        1.0, 30.0);
    config.species_cache_seconds = std::clamp(
        environment_number("SPECIES_CACHE_SECONDS", 8.0),
        1.0, 60.0);
    config.wildlife_test_captures_enabled =
        environment_flag("WILDLIFE_TEST_CAPTURES_ENABLED", false);
    config.wildlife_test_cooldown_seconds = std::clamp(
        environment_number("WILDLIFE_TEST_COOLDOWN_SECONDS", 30.0),
        5.0, 3600.0);
    config.wildlife_test_snapshot_retention =
        static_cast<std::size_t>(std::clamp(
            environment_number("WILDLIFE_TEST_SNAPSHOT_RETENTION", 200.0),
            1.0, 10000.0));
    const char* manual_storage_env = std::getenv("MANUAL_STORAGE_ROOT");
    if (manual_storage_env && *manual_storage_env) {
        config.manual_storage_root = manual_storage_env;
    }
    config.manual_recording_seconds = std::clamp(
        environment_number("MANUAL_RECORDING_SECONDS", 30.0), 5.0, 3600.0);
    config.manual_storage_max_gb = std::max(
        1.0, environment_number("MANUAL_STORAGE_MAX_GB", 200.0));
    config.manual_storage_min_free_gb = std::max(
        1.0, environment_number("MANUAL_STORAGE_MIN_FREE_GB", 10.0));
    config.reconnect_delay_seconds = std::clamp(
        environment_number("RECONNECT_DELAY_SECONDS", 2.0), 0.1, 60.0);
    const char* output_env = std::getenv("OUTPUT_DIR");
    config.output_dir = output_env && *output_env ? output_env : "runtime";
    config.snapshot_retention = static_cast<std::size_t>(std::max(
        1.0, environment_number("SNAPSHOT_RETENTION", 200.0)));
    const char* ntfy_server_env = std::getenv("NTFY_SERVER");
    const char* ntfy_topic_env = std::getenv("NTFY_TOPIC");
    config.ntfy_server = ntfy_server_env ? ntfy_server_env : "";
    config.ntfy_topic = ntfy_topic_env ? ntfy_topic_env : "";
    const char* driveway_label_env = std::getenv("DRIVEWAY_LABEL");
    const char* mailbox_label_env = std::getenv("MAILBOX_LABEL");
    const char* road_label_env = std::getenv("ROAD_LABEL");
    config.driveway_label = clean_zone_label(
        driveway_label_env ? driveway_label_env : "", "Driveway arrival");
    config.mailbox_label = clean_zone_label(
        mailbox_label_env ? mailbox_label_env : "", "Mailbox stop");
    config.road_label = clean_zone_label(
        road_label_env ? road_label_env : "", "Road archive");
    default_zones = current_zones_locked();
    std::filesystem::create_directories(config.output_dir / "events");
    std::filesystem::create_directories(config.output_dir / "road_events");
    std::filesystem::create_directories(config.output_dir / "wildlife_test");
    if (manual_storage_ready()) {
        std::filesystem::create_directories(manual_directory("snapshots"));
        std::filesystem::create_directories(manual_directory("videos"));
        enforce_manual_retention();
    }
    load_saved_zones();
    load_existing_events();

    const int configured_port = static_cast<int>(
        environment_number("WEB_PORT", 8000.0));
    const int port = argc > 1 ? std::max(1, std::atoi(argv[1]))
                              : std::max(1, configured_port);
    if (camera_url.empty()) {
        std::cerr << "CAMERA_URL is required\n";
        return 2;
    }

    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);

    std::thread camera_thread(capture_loop, camera_url);
    std::thread detector_thread(inference_loop);
    std::thread notification_thread(notification_loop);
    const int result = web_loop(port);

    running.store(false);
    notification_ready.notify_all();
    camera_thread.join();
    detector_thread.join();
    notification_thread.join();
    return result;
}
