#include "yoloV8.h"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
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
    float nms_threshold = 0.50f;
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
};

struct EventInfo {
    std::string id;
    std::string timestamp;
    std::string class_name;
    float confidence = 0.0f;
    std::string snapshot;
};

ServiceConfig config;

struct SharedState {
    std::mutex frame_mutex;
    cv::Mat latest_frame;
    std::vector<unsigned char> latest_jpeg;
    unsigned long jpeg_generation = 0;

    std::mutex status_mutex;
    bool camera_connected = false;
    double camera_fps = 0.0;
    double inference_ms = 0.0;
    double average_inference_ms = 0.0;
    unsigned long captured_frames = 0;
    unsigned long inference_count = 0;
    int relevant_objects = 0;
    double best_person_confidence = 0.0;
    bool person_in_zone = false;
    bool mailbox_vehicle_present = false;
    double mailbox_stationary_seconds = 0.0;
    unsigned long event_count = 0;
    unsigned long road_event_count = 0;
    unsigned long notification_count = 0;
    std::deque<EventInfo> recent_events;
    std::deque<EventInfo> recent_road_events;
    std::string last_error;
    std::string notification_error;
};

SharedState state;
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
    const std::string title = mailbox_event
        ? "Title: Driveway Watch — Mailbox"
        : "Title: Driveway Watch — " + display_class;
    const std::string tags = mailbox_event ? "Tags: mailbox_with_mail"
        : (event.class_name == "person" ? "Tags: bust_in_silhouette" : "Tags: car");
    std::ostringstream body;
    if (mailbox_event) {
        body << "Vehicle stopped near the mailbox (";
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

bool delete_archived_event(const std::string& event_id,
                           const std::string& archive_directory,
                           const std::string& log_filename,
                           bool road_archive) {
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
        auto& recent = road_archive ? state.recent_road_events : state.recent_events;
        for (auto iterator = recent.begin(); iterator != recent.end();) {
            if (iterator->id == event_id) {
                iterator = recent.erase(iterator);
            } else {
                ++iterator;
            }
        }
        auto& count = road_archive ? state.road_event_count : state.event_count;
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
    std::lock_guard<std::mutex> lock(state.status_mutex);
    state.recent_events = std::move(driveway.first);
    state.event_count = driveway.second;
    state.recent_road_events = std::move(road.first);
    state.road_event_count = road.second;
}

std::string status_json() {
    std::lock_guard<std::mutex> lock(state.status_mutex);
    std::ostringstream json;
    json << std::fixed << std::setprecision(1)
         << "{\"healthy\":" << (state.camera_connected ? "true" : "false")
         << ",\"camera_fps\":" << state.camera_fps
         << ",\"inference_ms\":" << state.inference_ms
         << ",\"average_inference_ms\":" << state.average_inference_ms
         << ",\"captured_frames\":" << state.captured_frames
         << ",\"inference_count\":" << state.inference_count
         << ",\"relevant_objects\":" << state.relevant_objects
         << ",\"best_person_confidence\":" << state.best_person_confidence
         << ",\"person_in_zone\":" << (state.person_in_zone ? "true" : "false")
         << ",\"mailbox_zone_enabled\":" << (config.mailbox_zone.empty() ? "false" : "true")
         << ",\"mailbox_vehicle_present\":" << (state.mailbox_vehicle_present ? "true" : "false")
         << ",\"mailbox_stationary_seconds\":" << state.mailbox_stationary_seconds
         << ",\"event_count\":" << state.event_count
         << ",\"road_event_count\":" << state.road_event_count
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
.video{overflow:hidden;border:1px solid #315c4b;border-radius:18px;background:#020504;box-shadow:0 20px 70px #0008}.video img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain}
.panel{margin:18px 0;padding:20px;border:1px solid #29483d;border-radius:16px;background:#102219dd}.panel h2{font-size:1.1rem;margin-bottom:16px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:14px;border-radius:12px;background:#183126}.label{font-size:.78rem;color:#91b7a5;text-transform:uppercase;letter-spacing:.06em}.value{font-size:1.25rem;margin-top:5px}
.tabs{display:flex;gap:8px;margin:18px 0}.tab{flex:1;padding:13px;border:1px solid #315c4b;border-radius:12px;background:#102219;color:#a9c7ba;font-weight:700;cursor:pointer}.tab.active{background:#24513d;color:#fff;border-color:#4d9a75}[hidden]{display:none!important}.events{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.event{overflow:hidden;border-radius:12px;background:#183126}.event-image{display:block;width:100%;padding:0;border:0;background:#07100d;cursor:zoom-in}.event-image img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}.meta{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px}.meta-text{min-width:0}.event strong{display:block;text-transform:capitalize}.event small{color:#91b7a5}.delete{display:grid;place-items:center;flex:0 0 42px;width:42px;height:42px;border:1px solid #70433f;border-radius:10px;background:#3b2422;color:#ffd4cf;font-size:1.15rem;cursor:pointer}.delete:hover{background:#60302c}.note{color:#a9c7ba;line-height:1.5}dialog.viewer{width:100vw;height:100vh;max-width:none;max-height:none;margin:0;padding:70px 18px 18px;border:0;background:#020504f2;color:#edf8f3}.viewer::backdrop{background:#000d}.viewer img{display:block;width:100%;height:calc(100vh - 110px);object-fit:contain}.viewer-bar{position:fixed;z-index:2;top:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 16px;background:#07100df2}.viewer-actions{display:flex;gap:8px}.viewer-close{display:grid;place-items:center;width:44px;height:44px;border:1px solid #527065;border-radius:10px;background:#183126;color:#fff;font-size:1.6rem;cursor:pointer}@media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header><h1>Driveway Watch</h1><div class="live"><span class="dot"></span><span id="health">Pi live</span></div></header>
<main>
<div class="video"><img src="/stream.mjpg" alt="Live driveway camera from Raspberry Pi"></div>
<p class="note" style="margin-top:10px">Blue: driveway arrival zone. Amber: mailbox stop zone. Purple: passing road traffic.</p>
<section class="panel"><h2>Raspberry Pi NCNN preview</h2><div class="stats">
<div class="stat"><div class="label">Camera</div><div class="value" id="camera">—</div></div>
<div class="stat"><div class="label">Detection</div><div class="value" id="inference">—</div></div>
<div class="stat"><div class="label">Objects now</div><div class="value" id="objects">—</div></div>
<div class="stat"><div class="label">Frames</div><div class="value" id="frames">—</div></div>
</div></section>
<div class="tabs" role="tablist" aria-label="Event archive"><button class="tab active" id="driveway-tab" type="button" role="tab" aria-selected="true">Driveway alerts (<span id="driveway-count">0</span>)</button><button class="tab" id="road-tab" type="button" role="tab" aria-selected="false">Road traffic (<span id="road-count">0</span>)</button></div>
<div id="driveway-panel" role="tabpanel"><section class="panel"><h2>Recent driveway events</h2><div class="events" id="events"><p class="note">No events yet</p></div></section><section class="panel"><h2>Notifications</h2><p class="note" id="notifications">Checking phone alerts…</p></section></div>
<div id="road-panel" role="tabpanel" hidden><section class="panel"><h2>Road traffic archive — no alerts</h2><p class="note" style="margin-bottom:16px">One snapshot is saved for each passing vehicle. These never send phone notifications.</p><div class="events" id="road-events"><p class="note">No passing vehicles captured yet</p></div></section></div>
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
document.getElementById('objects').textContent=s.relevant_objects;
document.getElementById('frames').textContent=s.captured_frames.toLocaleString();
document.getElementById('driveway-count').textContent=s.event_count.toLocaleString();
document.getElementById('road-count').textContent=s.road_event_count.toLocaleString();
const n=document.getElementById('notifications');
if(!s.notification_enabled)n.textContent='Phone alerts are not configured.';
else if(s.notification_error)n.textContent='Phone alerts need attention: '+s.notification_error;
else n.textContent='ntfy phone alerts are active. '+s.notification_count+' event alert'+(s.notification_count===1?' has':'s have')+' been sent since this service started.';}catch(e){document.getElementById('health').textContent='Reconnecting';}}
function renderEvents(list,box,emptyText){box.replaceChildren();if(!list.length){const p=document.createElement('p');p.className='note';p.textContent=emptyText;box.appendChild(p);return;}for(const event of list){const card=document.createElement('article');card.className='event';const imageButton=document.createElement('button');imageButton.className='event-image';imageButton.type='button';imageButton.setAttribute('aria-label','Open '+eventName(event)+' event full screen');const img=document.createElement('img');img.src=event.snapshot;img.alt=eventName(event)+' event';imageButton.appendChild(img);imageButton.addEventListener('click',()=>openEvent(event));const meta=document.createElement('div');meta.className='meta';const text=document.createElement('div');text.className='meta-text';const name=document.createElement('strong');name.textContent=eventName(event)+' — '+Math.round(event.confidence*100)+'%';const time=document.createElement('small');time.textContent=new Date(event.timestamp).toLocaleString();text.append(name,time);const remove=document.createElement('button');remove.className='delete';remove.type='button';remove.title='Delete this event';remove.setAttribute('aria-label','Delete '+eventName(event)+' event');remove.textContent='🗑️';remove.addEventListener('click',()=>deleteEvent(event));meta.append(text,remove);card.append(imageButton,meta);box.appendChild(card);}}
async function updateEvents(){try{const responses=await Promise.all([fetch('/api/events',{cache:'no-store'}),fetch('/api/road-events',{cache:'no-store'})]);const lists=await Promise.all(responses.map(response=>response.json()));renderEvents(lists[0],document.getElementById('events'),'No driveway events yet');renderEvents(lists[1],document.getElementById('road-events'),'No passing vehicles captured yet');}catch(e){}}
function selectArchive(road){document.getElementById('driveway-panel').hidden=road;document.getElementById('road-panel').hidden=!road;const drivewayTab=document.getElementById('driveway-tab');const roadTab=document.getElementById('road-tab');drivewayTab.classList.toggle('active',!road);roadTab.classList.toggle('active',road);drivewayTab.setAttribute('aria-selected',String(!road));roadTab.setAttribute('aria-selected',String(road));}
document.getElementById('driveway-tab').addEventListener('click',()=>selectArchive(false));document.getElementById('road-tab').addEventListener('click',()=>selectArchive(true));
update();updateEvents();setInterval(update,1000);setInterval(updateEvents,5000);
</script>
</body></html>)HTML";

bool relevant_class(int label) {
    return label == 0 || label == 1 || label == 2 || label == 3 ||
           label == 5 || label == 7;
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
        default: return "object";
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
            {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.camera_connected = false;
                state.last_error = "Could not open camera";
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(state.status_mutex);
            state.camera_connected = true;
            state.last_error.clear();
        }

        unsigned long interval_frames = 0;
        auto interval_start = Clock::now();
        cv::Mat frame;
        while (running.load() && capture.read(frame) && !frame.empty()) {
            {
                std::lock_guard<std::mutex> lock(state.frame_mutex);
                frame.copyTo(state.latest_frame);
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
        {
            std::lock_guard<std::mutex> lock(state.status_mutex);
            state.camera_connected = false;
            state.last_error = "Camera stream interrupted; reconnecting";
        }
    }
}

void inference_loop() {
    YoloV8 detector;
    detector.load(640);
    double total_inference_ms = 0.0;
    unsigned long event_sequence = 0;
    unsigned long road_event_sequence = 0;
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

    while (running.load()) {
        const auto cycle_start = Clock::now();
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(state.frame_mutex);
            if (!state.latest_frame.empty()) {
                state.latest_frame.copyTo(frame);
            }
        }
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            continue;
        }

        std::vector<Object> objects;
        const auto inference_start = Clock::now();
        detector.detect(frame, objects,
                        std::min(config.confidence_threshold,
                                 config.person_confidence_threshold),
                        config.nms_threshold);
        const double inference_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - inference_start).count();

        std::vector<cv::Point> zone_pixels;
        zone_pixels.reserve(config.driveway_zone.size());
        for (const auto& point : config.driveway_zone) {
            zone_pixels.emplace_back(
                static_cast<int>(point.x * frame.cols),
                static_cast<int>(point.y * frame.rows));
        }
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{zone_pixels}, true,
                      cv::Scalar(255, 190, 65), 2);

        if (!config.mailbox_zone.empty()) {
            std::vector<cv::Point> mailbox_pixels;
            mailbox_pixels.reserve(config.mailbox_zone.size());
            for (const auto& point : config.mailbox_zone) {
                mailbox_pixels.emplace_back(
                    static_cast<int>(point.x * frame.cols),
                    static_cast<int>(point.y * frame.rows));
            }
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{mailbox_pixels}, true,
                          cv::Scalar(0, 185, 255), 2);
            if (!mailbox_pixels.empty()) {
                cv::putText(frame, "MAILBOX STOP", mailbox_pixels.front() + cv::Point(4, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 185, 255), 2);
            }
        }

        if (!config.road_zone.empty()) {
            std::vector<cv::Point> road_pixels;
            road_pixels.reserve(config.road_zone.size());
            for (const auto& point : config.road_zone) {
                road_pixels.emplace_back(
                    static_cast<int>(point.x * frame.cols),
                    static_cast<int>(point.y * frame.rows));
            }
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{road_pixels}, true,
                          cv::Scalar(210, 80, 210), 2);
            if (!road_pixels.empty()) {
                cv::putText(frame, "ROAD ARCHIVE", road_pixels.front() + cv::Point(4, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(210, 80, 210), 2);
            }
        }

        int relevant_count = 0;
        double best_person_confidence = 0.0;
        bool person_in_zone = false;
        const Object* mailbox_vehicle = nullptr;
        cv::Point2f mailbox_vehicle_center;
        std::vector<RoadDetection> road_detections;
        std::map<std::string, const Object*> present_classes;
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
                : config.confidence_threshold;
            if (object.prob < required_confidence) {
                continue;
            }
            const cv::Point2f zone_point(
                (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                (object.rect.y + object.rect.height) / frame.rows);
            const bool in_mailbox = mailbox_vehicle_class(object.label) &&
                !config.mailbox_zone.empty() &&
                cv::pointPolygonTest(config.mailbox_zone, zone_point, false) >= 0;
            if (in_mailbox && (!mailbox_vehicle || mailbox_vehicle->prob < object.prob)) {
                mailbox_vehicle = &object;
                mailbox_vehicle_center = cv::Point2f(
                    (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                    (object.rect.y + object.rect.height * 0.5f) / frame.rows);
            }
            const bool in_road = road_vehicle_class(object.label) &&
                !config.road_zone.empty() &&
                cv::pointPolygonTest(config.road_zone, zone_point, false) >= 0;
            if (in_road) {
                road_detections.push_back({&object, cv::Point2f(
                    (object.rect.x + object.rect.width * 0.5f) / frame.cols,
                    (object.rect.y + object.rect.height * 0.5f) / frame.rows)});
            }
            const bool in_driveway =
                cv::pointPolygonTest(config.driveway_zone, zone_point, false) >= 0;
            if (!in_driveway) {
                if (in_mailbox || in_road) {
                    ++relevant_count;
                    const cv::Rect box = object.rect;
                    const cv::Scalar color = in_mailbox
                        ? cv::Scalar(0, 185, 255) : cv::Scalar(210, 80, 210);
                    cv::rectangle(frame, box, color, 3);
                    std::ostringstream label;
                    label << (in_mailbox ? "mailbox " : "road ")
                          << class_name(object.label) << " " << std::fixed
                          << std::setprecision(0) << object.prob * 100.0f << "%";
                    cv::putText(frame, label.str(),
                                cv::Point(box.x, std::max(24, box.y) - 7),
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2);
                }
                continue;
            }
            if (object.label == 0) {
                person_in_zone = true;
            }
            ++relevant_count;
            const std::string name = class_name(object.label);
            const auto existing = present_classes.find(name);
            if (existing == present_classes.end() ||
                existing->second->prob < object.prob) {
                present_classes[name] = &object;
            }
            const cv::Rect box = object.rect;
            cv::rectangle(frame, box, cv::Scalar(65, 229, 141), 3);
            std::ostringstream label;
            label << name << " " << std::fixed
                  << std::setprecision(0) << object.prob * 100.0f << "%";
            const int top = std::max(24, box.y);
            cv::putText(frame, label.str(), cv::Point(box.x, top - 7),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(65, 229, 141), 2);
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
            EventInfo road_event = create_road_event(
                frame, class_name(detection.object->label), detection.object->prob,
                ++road_event_sequence);
            {
                std::lock_guard<std::mutex> lock(state.status_mutex);
                state.recent_road_events.push_front(std::move(road_event));
                while (state.recent_road_events.size() > 20) {
                    state.recent_road_events.pop_back();
                }
                ++state.road_event_count;
            }
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
            if (!mailbox_alerted_this_visit &&
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
        for (const auto& [name, object] : present_classes) {
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
            EventInfo event = create_event(frame, name, object->prob, ++event_sequence);
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

        std::vector<unsigned char> jpeg;
        cv::imencode(".jpg", frame, jpeg, jpeg_options);
        {
            std::lock_guard<std::mutex> lock(state.frame_mutex);
            state.latest_jpeg = std::move(jpeg);
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

        const auto next_cycle = cycle_start + std::chrono::milliseconds(200);
        std::this_thread::sleep_until(next_cycle);
    }
}

void serve_mjpeg(int client_socket) {
    const std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n";
    if (!send_all(client_socket, headers.data(), headers.size())) {
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
                jpeg = state.latest_jpeg;
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

void handle_client(int client_socket) {
    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    char request_buffer[4096]{};
    const ssize_t received = recv(client_socket, request_buffer,
                                  sizeof(request_buffer) - 1, 0);
    if (received <= 0) {
        close(client_socket);
        return;
    }

    std::istringstream request(std::string(request_buffer, received));
    std::string method;
    std::string path;
    request >> method >> path;

    if (method == "DELETE" && path.rfind("/api/road-events/", 0) == 0) {
        const std::string event_id = path.substr(std::strlen("/api/road-events/"));
        if (delete_road_event(event_id)) {
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
    } else if (path == "/stream.mjpg") {
        serve_mjpeg(client_socket);
    } else if (path.rfind("/events/", 0) == 0) {
        const std::string filename = path.substr(std::strlen("/events/"));
        serve_archived_image(client_socket, "events", filename);
    } else if (path.rfind("/road-events/", 0) == 0) {
        const std::string filename = path.substr(std::strlen("/road-events/"));
        serve_archived_image(client_socket, "road_events", filename);
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
    config.nms_threshold = static_cast<float>(std::clamp(
        environment_number("NMS_THRESHOLD", 0.50), 0.01, 0.99));
    const char* output_env = std::getenv("OUTPUT_DIR");
    config.output_dir = output_env && *output_env ? output_env : "runtime";
    config.snapshot_retention = static_cast<std::size_t>(std::max(
        1.0, environment_number("SNAPSHOT_RETENTION", 200.0)));
    const char* ntfy_server_env = std::getenv("NTFY_SERVER");
    const char* ntfy_topic_env = std::getenv("NTFY_TOPIC");
    config.ntfy_server = ntfy_server_env ? ntfy_server_env : "";
    config.ntfy_topic = ntfy_topic_env ? ntfy_topic_env : "";
    std::filesystem::create_directories(config.output_dir / "events");
    std::filesystem::create_directories(config.output_dir / "road_events");
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
