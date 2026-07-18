# Acknowledgements

AI Driveway Watch grew from a real household camera experiment into a reusable
local-first project through hands-on testing, open-source engineering, and
human–AI pair programming.

## Qengineering

The production detector uses the YOLOv8/NCNN C++ implementation and converted
YOLOv8n model assets from
[Qengineering/YoloV8-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4).
Qengineering’s practical Raspberry Pi optimization work made the fast,
all-local inference path possible. Those upstream files remain covered by the
BSD 3-Clause license included by the pinned asset installer.

Neither Qengineering nor its contributors endorse this project; the credit is
gratitude and required attribution for their upstream work.

## OpenAI Codex

This project was developed collaboratively by `prichardsondev` and OpenAI
Codex. Codex assisted with C++ implementation, Raspberry Pi deployment,
performance verification, dashboard design, privacy review, testing, and
documentation. The human owner supplied the goals, physical-world context,
camera calibration, acceptance testing, and final product decisions.

That collaboration is also why this repository includes `AGENTS.md` and a set
of reusable prompts: the goal is to make the project understandable and safely
adaptable by both people and their coding agents.

OpenAI does not sponsor or endorse this project.
