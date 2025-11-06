# Phase 1 — Project Charter
**Project:** Redefining Recovery Systems for Eating Disorders (University Health Services)  
**Prepared by:** Reese Yoon  
**Date:** November 06, 2025

## 1) Problem Statement (Client-Framed)
University Health Services needs a scalable, data-informed way to understand students’ barriers, preferences, and outcomes related to eating-disorder recovery so they can design better services, increase engagement, and measure impact. Current feedback is scattered, making prioritization difficult.

## 2) Objectives
- **O1 – Insight Generation:** Convert raw survey/interview data into prioritized needs and opportunity areas.
- **O2 – Experience Design:** Produce personas, journey maps, and an experience vision aligned with student/clinician goals.
- **O3 – Operational Excellence:** Deliver a working analytics dashboard (v1) that reduces manual analysis time.
- **O4 – Cloud Readiness:** Host a secure, shareable prototype on cloud (IBM Cloud or AWS) to simulate client delivery.

## 3) Success Metrics (SMART)
- **Adoption/Engagement:** +20% student interactions with recovery resources within one term.
- **Efficiency:** Synthesis time per batch down **70%** (3h → < 1h).
- **Insight Quality:** Stakeholder satisfaction ≥ **4.5/5** on “insight usefulness”.
- **Reliability:** Dashboard uptime ≥ **99%** over 30 days (when deployed).
- **Data Coverage:** ≥ **90%** responses categorized by topic/sentiment models (later phases).

## 4) Scope
**In-Scope (Phase 1)**
- Import and audit **current** survey CSVs only (no external data).
- Light PII scan and privacy plan.
- Initial backlog creation and risk log.
- Discovery plan + interview guide (students, clinicians, admin).

**Out-of-Scope (Phase 1)**
- Clinical recommendations, EHR integrations, mobile-native builds.

## 5) Stakeholders & Users
- **Primary:** Students in recovery; Clinicians (therapists, dietitians)
- **Secondary:** University admins; IT/privacy; counseling center staff

## 6) Risks & Mitigations
- **Data Sensitivity:** Use de-identified datasets; access controls; privacy review.
- **Sampling Bias:** Document bias; plan stratified analysis; caveat insights.
- **Scope Creep:** MoSCoW priorities; timebox features; maintain backlog.

## 7) Governance & Ethics
- No PII at rest; remove/obfuscate personal identifiers.
- Document data lineage and transformations.
- Outputs inform **program design**, not clinical decisions.

## 8) Deliverables (Phase 1 → Phase 2)
- Project Charter (this doc)
- Stakeholder Map + RACI
- Discovery Plan & Interview Guide
- Initial Requirements Backlog
- Data Audit Report (from `scripts/audit_data.py`)

## 9) IBM Alignment
- **Application Consultant:** data pipeline, backlog, measurable efficiency.
- **Experience Consultant:** personas, journeys, experience principles, workshops.

