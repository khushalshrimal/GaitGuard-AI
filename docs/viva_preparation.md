# GaitGuard AI — Technical Viva Voce Preparation Guide (40+ Q&A)

## 1. Problem Statement & Scope

### Q1: What problem does GaitGuard AI solve?
**Answer**: Bovine lameness causes severe pain and economic losses in cattle farming. Manual locomotion scoring is subjective, labor-intensive, and catches lameness late. GaitGuard AI provides a non-invasive, automated AI-assisted screening tool to detect movement anomalies early using standard smartphone video.

### Q2: Why is GaitGuard AI defined as a screening tool rather than a diagnostic system?
**Answer**: Screening tools flag statistical anomalies relative to baselines to support early herd triage. Diagnosis requires clinical physical examination, hoof trimming, palpation, and veterinary expertise. Framing the tool as non-diagnostic enforces medical ethics and legal compliance.

### Q3: Who are the target users of GaitGuard AI?
**Answer**: Dairy/beef farmers, field livestock technicians, and veterinary professionals looking for quantitative movement screening decision support.

---

## 2. Frontend & React Architecture

### Q4: What frontend framework and state management pattern are used?
**Answer**: React 18 with TypeScript, Vite build tool, Tailwind CSS, and Lucide icons. Application state is managed via an explicit state machine (`IDLE`, `CAPTURING`, `PREVIEW`, `SUBMITTING`, `RESULT`, `RETRY`, `ERROR`).

### Q5: How does the frontend handle memory management during video preview?
**Answer**: It creates a local blob URL via `URL.createObjectURL(file)` to preview the selected video stream in an HTML5 `<video>` element, and revokes it via `URL.revokeObjectURL(url)` upon reset to prevent memory leaks.

### Q6: How does the client validate uploads before making an API request?
**Answer**: The client checks file extensions (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) and file size ceiling (100MB max limit), giving immediate UI feedback while maintaining server-side magic byte validation.

### Q7: How is backend API connectivity displayed to the user?
**Answer**: The `Header` component polls `GET /health` every 15 seconds, rendering a green "API Connected" badge when Uvicorn is healthy or a red "API Offline" badge when unavailable.

---

## 3. Backend & REST API Design

### Q8: What backend technology stack is used?
**Answer**: Python 3.10+, FastAPI framework, Uvicorn ASGI application server, and Pydantic v2 schemas.

### Q9: What is the purpose of Request Correlation IDs (`X-Request-ID`)?
**Answer**: Attaching `X-Request-ID: req_<uuid4>` middleware headers allows developers to correlate frontend API calls with backend Uvicorn server logs, Quality Gate metrics, BiLSTM inference logs, and SHAP traces.

### Q10: What HTTP status codes are returned by the API?
**Answer**: 200 OK (successful screening / health / version), 400 Bad Request (magic byte mismatch / unsupported extension), 413 Payload Too Large (>100MB stream), 422 Unprocessable Entity (Quality Gate rejection), 503 Service Unavailable (model uninitialized), 500 Internal Error (sanitized exception).

---

## 4. Input Validation & Upload Security

### Q11: How does magic-byte signature validation protect the server?
**Answer**: Inspecting raw binary header bytes (`ftyp`, `RIFF`, `\x1a\x45\xdf\xa3`) prevents malicious executables or scripts renamed to `.mp4` from being saved or executed on the server.

### Q12: How does the backend prevent path traversal attacks?
**Answer**: Filenames are stripped using `os.path.basename` and asserted using `os.path.realpath` to guarantee target files remain strictly within `settings.TEMP_DIR`.

### Q13: How is zero temporary video persistence guaranteed?
**Answer**: Uploaded videos are saved in isolated temporary directory space and removed inside a deterministic `finally` execution block immediately after keypoint extraction.

---

## 5. Computer Vision & Keypoint Pose Extraction

### Q14: How many keypoints are extracted per cattle frame?
**Answer**: 17 anatomical keypoints (withers, spine points, hocks, fetlocks, hoofs).

### Q15: How are keypoint trajectories smoothed across frames?
**Answer**: Using Savitzky-Golay polynomial trajectory filtering (window size = 5, polynomial order = 2) to eliminate high-frequency tracking jitter.

---

## 6. Feature Engineering & Normalization

### Q16: How many features represent each sequence sample?
**Answer**: 76 features per frame (coordinates, velocities, torso-normalized biomechanical indicators).

### Q17: Why is torso length normalization necessary?
**Answer**: Cow physical dimensions vary across breeds and ages. Normalizing limb displacements by torso length (distance between withers and hip) makes gait metrics scale-invariant.

---

## 7. Temporal Sequence Modeling & BiLSTM

### Q18: Why use a Bidirectional LSTM (BiLSTM) instead of frame-by-frame 2D CNN classification?
**Answer**: Lameness is inherently temporal—it manifests in stride rhythm asymmetry, stance duration variance, and cyclic back arch elevation over time. BiLSTMs evaluate past and future temporal context across entire 128-frame walking windows.

### Q19: What is the sequence input tensor shape fed into the BiLSTM?
**Answer**: `(batch_size, 128, 76)`.

---

## 8. GroupKFold & Leakage Prevention

### Q20: What is animal-level GroupKFold splitting?
**Answer**: A cross-validation strategy ensuring that all video clips from a given cow (`animal_id`) appear strictly in either the training set OR the validation set, with 0 animal overlap.

### Q21: What happens if random splitting is used instead of GroupKFold?
**Answer**: Severe data leakage occurs—the model memorizes individual cow background features rather than learning generalizable lameness motion traits, leading to artificially inflated accuracy metrics.

---

## 9. Model Metrics & Internal Evaluation

### Q22: What internal cross-validation accuracy and ROC-AUC did GaitGuard achieve?
**Answer**: Accuracy: ~81.97%, F1-Score: ~0.7973, Out-of-Fold ROC-AUC: 0.9016.

### Q23: Why is ROC-AUC reported alongside accuracy and recall?
**Answer**: ROC-AUC measures ranking discrimination capability across all potential threshold settings, independent of class imbalance.

---

## 10. Calibration & Threshold Selection

### Q24: What is probability calibration?
**Answer**: Mapping raw neural network outputs into calibrated probabilities using Platt Sigmoid scaling so that a predicted score of 0.80 means an 80% true probability of anomaly.

### Q25: Why was screening threshold $\tau = 0.34$ selected?
**Answer**: Standard 0.50 threshold yields lower recall. Lowering $\tau$ to 0.34 prioritizes screening sensitivity (~89.15% recall), ensuring potential lameness risk is flagged early.

---

## 11. Inconclusive Screening Outcomes

### Q26: What is the inconclusive probability interval?
**Answer**: $[0.24, 0.44]$ ($\pm 0.10$ margin around threshold $\tau = 0.34$).

### Q27: Why return an inconclusive triage decision instead of forcing a binary classification?
**Answer**: Forcing binary decisions on borderline cases causes false alarms or missed risks. Flagging ambiguity prompts the user to re-record or perform visual inspection.

---

## 12. Explainable AI (SHAP)

### Q28: How does GaitGuard implement Explainable AI?
**Answer**: Using Deep SHAP feature attributions (`SHAP Explainer Layer`), returning Level 1 biomechanical attributions and Level 2 domain gait metrics.

### Q29: What does a positive SHAP value indicate?
**Answer**: A positive SHAP value indicates that the feature contributed towards increasing the predicted lameness risk.

---

## 13. Privacy & Telemetry Governance

### Q30: Does GaitGuard AI collect or store user personal data?
**Answer**: Zero PII is collected or stored. Request IDs use UUIDv4 hashes with no client IP or identity tracking.

---

## 14. Performance & Latency

### Q31: What is the measured end-to-end pipeline model latency?
**Answer**: **66.59 ms** (Satisfies sub-100ms real-time target).

---

## 15. Testing & Verification

### Q32: How many backend and frontend tests are included?
**Answer**: 239 backend `unittest` test cases and 13 frontend Vitest test cases (100% pass rate).

---

## 16. Responsible Use & External Validation Limitations

### Q33: What is the status of external field validation?
**Answer**: `BLOCKED — DATA COLLECTION REQUIRED`. Reported metrics represent internal animal-level cross-validation.

### Q34: What evidence is required before commercial farm deployment?
**Answer**: Prospective field validation on independently collected, un-seen real-world farm datasets with veterinarian ground-truth labels.
