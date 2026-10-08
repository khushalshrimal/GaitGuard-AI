# GaitGuard AI — Phase 14 Reproducibility Specification

## Environment Details
- **Operating System**: Windows (AMD64)
- **Python Version**: 3.14.0
- **PyTorch Version**: 2.6.0+cpu
- **NumPy Version**: 2.2.2
- **Pandas Version**: 2.2.3
- **Scipy Version**: 1.15.1
- **Random Seed**: `42`

## System Checksums & Hashes
- `docs/phase14_frozen_configuration.json`: Validated
- `validation/field_validation_manifest.csv`: Validated
- `gaitguard/config.py`: Validated

## Execution Command
```powershell
$env:PYTHONPATH="." ; py -3 scripts/run_phase14_validation.py
$env:PYTHONPATH="." ; py -3 scripts/analyze_phase14_domain_shift.py
$env:PYTHONPATH="." ; py -3 scripts/analyze_phase14_robustness.py
$env:PYTHONPATH="." ; py -3 scripts/analyze_phase14_errors.py
```
