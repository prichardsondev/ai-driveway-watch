# Architecture

```text
RTSP/RTSPS camera
       |
       v
OpenCV 5 capture thread ------> latest frame
                                  |
                                  v
                         YOLOv8n / NCNN
                                  |
              +-------------+-------------+-------------+
              |             |             |             |
              v             v             v             v
       driveway state  mailbox dwell  animal union  road tracker
              |             |             |             |
              +-------------+-------------+             |
                            |                           |
                            v                           v
                   alert event archive         vehicle-only road archive
                            |                           |
                            v                           v
                       ntfy queue               dashboard gallery
```

## Threads

- capture: continuously decodes the camera and replaces the latest frame;
- inference: samples the latest frame five times per second and evaluates all
  zone rules;
- notification: sends accepted driveway/mailbox/animal messages without blocking
  inference;
- HTTP: serves the MJPEG stream, status, galleries, images, and deletion APIs.

## Event stores

Driveway and mailbox events use `runtime/events/` plus `runtime/events.csv`.
Animal events from any boundary use the same alert store and one shared animal
cooldown to avoid repeated or oscillating species guesses for a lingering animal.
Passing traffic uses `runtime/road_events/` plus `runtime/road_events.csv`.
Only the newest entries are held in memory for the dashboard; CSV history and
JPEGs persist across service restarts.

## Zone semantics

Polygons use normalized coordinates, so one configuration survives resolution
changes. Membership is based on the detection box’s bottom-center point rather
than its geometric center.

## Road tracking

Road detections are matched to recent tracks by normalized center distance.
A new unmatched track creates one snapshot. Tracks expire quickly after a
vehicle disappears, making the rule independent of direction while suppressing
duplicate frames of the same vehicle.
