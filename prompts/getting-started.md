# Prompt: get started with AI Driveway Watch

I want to set up AI Driveway Watch from this repository on my Raspberry Pi.

Start by checking whether the repository is already present. If it is not,
clone it from:

`https://github.com/prichardsondev/ai-driveway-watch`

## Before you start: make sure you have these things

Make sure you have the following ready before asking the agent to begin:

- A Raspberry Pi (Pi 5 recommended) with Raspberry Pi OS, power, and network
  access.
- A PoE-capable camera, a PoE injector or PoE switch, and an Ethernet cable.
  The camera and Pi should be on the same local network during setup.
- A laptop or desktop on that network so you can open the dashboard and run
  setup commands.
- Optional but recommended: a USB 3 SSD for saved snapshots and recordings.
- The camera's local username and password, kept private and entered only on
  the Pi when prompted.

If you do not have every item yet, tell me what you do have and I will explain
what can be tested safely before we continue. Do not buy hardware or format
storage without first confirming the exact model and disk.

Then walk me through the setup one step at a time. Do not assume I already
know Linux, Raspberry Pi services, OpenCV, NCNN, RTSP, or Tailscale. Pause for
my confirmation after any physical action, reboot, format, camera setting, or
network change.

Use this order:

1. Identify the Pi, storage, operating system, and available hardware without
   changing anything.
2. Explain what camera connection information is needed and help me enable
   RTSP/ONVIF on the camera without asking me to paste a password or camera
   URL into chat.
3. Install or verify the documented OpenCV 5, NCNN, and build dependencies.
4. Configure a private `.env` from `.env.example`, keeping camera URLs,
   credentials, notification topics, LAN addresses, and event images local.
5. Build the C++ NCNN service and run a safe read-only health check.
6. Help me draw and save driveway, mailbox, and road boundaries from the
   dashboard.
7. Install the systemd service only after the manual run is healthy.
8. Verify the camera stream, status API, event APIs, notifications, storage,
   temperature, and throttling without manufacturing an event.

Explain each command before running it. Never print `.env` values, secrets,
private camera images, or household identifiers. Never format a disk, delete
archives, expose a port to the internet, or change router settings without my
explicit approval. If something fails, diagnose it and preserve a rollback
path before changing the production service.
