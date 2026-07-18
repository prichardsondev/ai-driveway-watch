# Security and privacy

## Supported deployment

AI Driveway Watch is intended for a trusted home LAN. The dashboard has no
built-in user authentication and must not be exposed directly to the internet.

For remote viewing, place it behind an authenticated private network. Do not
configure router port forwarding to the dashboard.

## Secrets

Keep these values only in an ignored, permission-restricted `.env` on the
target device:

- RTSP/RTSPS camera URLs;
- ntfy topics or authentication material;
- private hostnames and addresses when they reveal household infrastructure.

Before publishing a change, scan all candidate files for URLs, addresses,
topics, event images, CSV logs, and personal names.

## Event data

Snapshots may contain people, vehicles, plates, addresses, and timestamps.
The `runtime/` tree is ignored by Git and should be protected like camera
footage. Configure retention for the household’s needs and applicable law.

## Reporting a vulnerability

Do not open a public issue containing a camera URL, topic, snapshot, or exploit
details tied to a live home installation. Contact the repository owner
privately through their GitHub profile first.
