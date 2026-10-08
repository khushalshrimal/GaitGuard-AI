# GaitGuard AI — Phase 15 Data Provenance Protocol

## 1. Provenance Record Requirements
Every external validation video file indexed in `validation/external_validation_manifest.csv` must record full data lineage:
- **Source Identifier**: Unique farm/site origin code (e.g. `FARM_ALPHA`).
- **Acquisition Timestamp**: ISO 8601 recording date/time.
- **Original Filename**: Original raw camera file name.
- **SHA-256 Checksum**: Cryptographic file checksum verifying integrity.
- **Recording Camera Device**: Camera hardware make/model and resolution.
- **Consent Status**: Farm owner research participation consent.

## 2. Integrity Verification
Cryptographic file hashes prevent accidental file corruption, file duplication, or dataset tampering across multi-site field trial evaluations.
