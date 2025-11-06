# Data Audit Report — Phase 1

**Date:** Nov 6th 2025  
**Owner:** Reese Yoon

## Files Audited (data/raw/)
- Survey for Medical professionals (Responses) - Form Responses 1.csv
- Help Us Build a Better ED Recovery System  (Responses) - Form Responses 1.csv

## Summary
- **Null values (total across files):** 159
- **Empty strings (total across files):** 0
- **Potential PII columns flagged:** “If you said Yes, Please leave your email down below”


## Notes
- Encoding handled with UTF-8 and latin-1 fallback.
- Next action: Exclude any PII-like columns from downstream processing or mask before use.

## Decision
- Proceed using only the **current CSVs** in `data/raw/`.
- If any columns are flagged as PII, create a masked copy in `data/processed/` (Phase 2).
