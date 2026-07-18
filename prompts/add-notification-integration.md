# Prompt: add a notification integration

Use this repository’s `AGENTS.md` privacy and deployment rules.

Add a new notification subscriber or publisher without coupling network I/O to
camera capture or inference.

Requirements:

1. driveway and mailbox events may notify; road archive events must not;
2. all credentials stay in the private `.env` or a protected service file;
3. delivery must run through a bounded queue and time out cleanly;
4. duplicate event cooldowns remain authoritative;
5. the dashboard reports enabled state and delivery errors without secrets;
6. include a harmless text-only test path;
7. document how to disable and roll back the integration.
