# GaitGuard AI — Privacy, Security & Data Protection Audit Report

## 1. Executive Summary

GaitGuard AI is engineered around **Privacy-by-Design** and **Zero-Persistence Data Retention** principles. Video files uploaded for gait analysis are processed in isolated operating system temporary directory space (`TEMP_DIR`), analyzed strictly in memory during keypoint extraction, and guaranteed to be deleted immediately after inference finishes.

Zero raw animal videos, frame images, farm GPS coordinates, or owner PII (Personally Identifiable Information) are retained on disk or sent to external server databases.

---

## 2. Temporary Data Lifecycle & Guaranteed Cleanup

### 2.1 File Storage & Path Traversal Containment
* **Storage Location**: Uploaded video files are saved to `app.config.settings.TEMP_DIR` using standard Python `tempfile` naming convention (`gaitguard_upload_<uuid4>.mp4`).
* **Path Containment**: Filenames are sanitized via `os.path.basename` and asserted using `os.path.realpath` to prevent directory traversal attacks (`../../etc/passwd`).

### 2.2 Execution Path Audit & Deletion Guarantees

| Execution Flow | Endpoint Outcome | Cleanup Mechanism | Audit Result |
| :--- | :--- | :--- | :--- |
| **Successful Screening** | HTTP 200 OK | `finally` execution block in `app/routes/inference.py` | 0 Leaks (PASS) |
| **Quality Gate Rejection** | HTTP 200 / 422 Retry | `finally` execution block executes before response return | 0 Leaks (PASS) |
| **Magic Byte Mismatch** | HTTP 400 Bad Request | File deleted immediately inside `VideoService.save_temp_file()` | 0 Leaks (PASS) |
| **File Exceeds 100MB Cap** | HTTP 413 Payload Too Large | Partial stream aborted; deleted in `finally` block | 0 Leaks (PASS) |
| **Unhandled Exception** | HTTP 500 Internal Error | Outer `try...finally` block guarantees cleanup | 0 Leaks (PASS) |

---

## 3. Input Validation & Magic Byte Security

To prevent malicious file upload attacks (e.g. executable scripts or binary exploits renamed with a `.mp4` extension):
1. **Extension Check**: Filename extension verified against allowed whitelist (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
2. **Magic Byte Signature Inspection**: Server reads initial 64 bytes of video file stream and verifies valid video magic signatures:
   - `ftyp` (MP4 / MOV)
   - `RIFF` (AVI)
   - `\x1a\x45\xdf\xa3` (WebM / MKV)

---

## 4. Telemetry & Information Governance

* **Zero PII Collection**: GaitGuard AI collects zero user names, phone numbers, email addresses, or GPS coordinates.
* **Cryptographic Request IDs**: Request IDs (`req_<uuid4_hex>`) are randomly generated UUIDv4 hashes containing no client IP address or user identification information.
* **Traceback Shielding**: Production exception handlers capture full stack trace details in internal logs tagged with Request IDs while returning generic safe messages to clients.

---

## 5. Non-Diagnostic Medical Boundary

All API responses and UI screens display the following non-diagnostic disclaimer:
> *"GaitGuard AI provides AI-assisted cattle gait and lameness-risk screening. It is not a veterinary diagnosis and does not replace assessment by a qualified veterinarian."*
