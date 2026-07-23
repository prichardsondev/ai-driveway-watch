# Prompt: diagnose a missed detection

Use this repository’s `AGENTS.md` privacy and deployment rules.

A real person, vehicle, or animal crossed one of my configured zones but did
not create the expected event. Diagnose the cause without changing thresholds
first.

Check, in order:

1. camera continuity and actual inference rate;
2. the best model confidence, including detections outside the target polygon;
   for animals also verify `wildlife_model_ready`, its inference rate, and its
   independent wildlife threshold;
3. bottom-center membership in the correct polygon;
4. active-state, clear-time, track, dwell, and cooldown suppression;
5. vehicle class confusion, or fallback to COCO animal classes if the
   MegaDetector model did not load;
6. notification delivery only after confirming the event was created.

Report the evidence and smallest safe tuning change. Do not expose `.env`, the
camera URL, the ntfy topic, or private images.
