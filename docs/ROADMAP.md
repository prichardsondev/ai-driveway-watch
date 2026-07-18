# Roadmap

## Interactive boundary editor — first version complete

The dashboard now provides a phone-friendly editor over the live frame:

- select driveway, mailbox, or road polygon;
- redraw polygons by tapping corners and undo the latest point;
- preview without changing live detection;
- Save and Cancel controls;
- validate normalized coordinates server-side;
- write a dedicated zone configuration file atomically;
- reload zones without restarting camera capture;
- restore the immutable startup calibration from `.env`.

Planned editor refinements:

- drag existing handles and insert or remove individual points;
- export/import a redacted configuration that never includes secrets.

## Later possibilities

- optional authenticated remote access documentation;
- event filters and date-based archive browsing;
- disk-usage display and retention by age or storage budget;
- kitchen speaker subscriber for local spoken announcements;
- notification attachments when using a private, authenticated ntfy server;
- improved multi-object tracking for dense traffic.
