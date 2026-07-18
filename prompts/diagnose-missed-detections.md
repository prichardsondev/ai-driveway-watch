# Prompt: diagnose a missed detection

Use this repository’s `AGENTS.md` privacy and deployment rules.

A real person or vehicle crossed one of my configured zones but did not create
the expected event. Diagnose the cause without changing thresholds first.

Check, in order:

1. camera continuity and actual inference rate;
2. the best model confidence, including detections outside the target polygon;
3. bottom-center membership in the correct polygon;
4. active-state, clear-time, track, dwell, and cooldown suppression;
5. class confusion between car, truck, bus, motorcycle, and bicycle;
6. notification delivery only after confirming the event was created.

Report the evidence and smallest safe tuning change. Do not expose `.env`, the
camera URL, the ntfy topic, or private images.
