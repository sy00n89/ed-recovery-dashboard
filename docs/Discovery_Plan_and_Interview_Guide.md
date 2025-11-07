# Phase 2
## Executive Summary
- Business need: Universities need earlier, scalable ED support that complements care pathways.
- Users: Students with lived ED experience; clinicians/dietitians who triage and recommend care.
- Phase 2 goal: Turn “wishes” into prioritized experience hypotheses and quantify treatment gaps vs pro recommendations.
- Key outcomes to date:
  - 80%+ of student “wishes” classified into non-Other themes after auto-promotion.
  - Top wish themes: Listen/Validate, Identity-specific support, Meal support/Practical help, Peer community.
  - Gaps vs pros: higher student demand for Listen/Validate + identity support; pros emphasize peer/community and education/resources.
- Phase 3 success metric (forward-looking): ≥60% intent to use at least one proposed feature within 1 week; ≥30% week-2 re-use intent.

## Phase 2 Plan 
- Methods: Live Google Sheets → Streamlit dashboard; text normalization; theme bucketing; n-gram mining + auto-promotion to reduce “Other”; side-by-side gap deltas; quote sampling.
- Artifacts: Sections D/E/F screenshots; gap table with deltas; three hypotheses (below).
- Top hypotheses:
  1. “Listen Mode” micro-feature: just-in-time prompts that validate without “just eat.”
  2. Identity-aligned peer nudge: small, moderated circles (men/ARFID/LGBTQ+) surfaced at the right moment.
  3. Meal assist micro-journeys: concrete, low-friction meal tasks + celebratory feedback.
- Risks/guards: Crisis routing, anonymity, clinician configurable guardrails.

## Interview Guide
- Hypothesis probes aligned to D/E themes (you already have these—promote them to the top).
- Adoption test: “If we shipped Listen Mode, would you use it this week? What would stop you?”
- Clinician feasibility: “What minimal signals/data would let you recommend this safely?”
- Decision rules: Say you’ll greenlight a concept if ≥60% of students accept and clinicians name no high-risk contraindications without mitigation.

# Phase 1 — Discovery & Data Collection Summary

## Purpose
The goal of Phase 1 was to understand the recovery experience and professional perspective around eating disorders through structured surveys. Insights from this phase shape the foundation for Mirror’s app design and recovery model.

---

## What Has Been Completed

### 1. Data Collection
- **Two separate Google Forms** were distributed:
  - *Help Us Build a Better ED Recovery System* (Students/Individuals)
  - *Survey for Medical Professionals*
- Each form targeted unique but complementary insights:
  - Students shared experiences, challenges, and desired features.
  - Professionals shared treatment approaches, resource recommendations, and observed barriers.

### 2. Data Processing
- All responses sync automatically to public Google Sheets.  
- Streamlit pulls those sheets via `export?format=csv&gid=...` for live updates.  
- PII columns are automatically removed, and columns are inferred via keyword matching.  
- Text responses are split and normalized for keyword analysis.

### 3. Analysis & Dashboard Creation
- **Dashboard Overview (Phase 1)**
  - **Section A – Overview:** Age/gender demographics and participant breakdowns.  
  - **Section B – Barriers to Recovery:** Combined keyword-based visualization from both groups.  
  - **Section C – Helpful Supports & Alignment:** Comparison of what individuals found helpful vs. what professionals provide.
- **Methodology**
  - Implemented lightweight NLP keyword mapping for “barriers” and “supports.”  
  - Used `plotly.express` for clear visual storytelling (Top-N limit for readability).  
  - Automatic refresh every 60 seconds.

### 4. Key Learnings (Phase 1)
- Cost, access, and stigma remain dominant barriers across both groups.  
- Professionals highlight structured therapies (CBT, DBT, FBT), while individuals emphasize peer and safe-space support.  
- There is an alignment gap between the clinical focus and the interpersonal/community support individuals desire.  
- The system now provides an at-a-glance summary without manual setup.

---

## Interview Guide (for Qualitative Validation)

**Purpose:** To validate Phase 1 survey insights and explore “why” behind the quantitative trends.

### 1. For Individuals
1. What aspects of support (formal or informal) have helped you the most in recovery?  
2. How do you perceive cost, access, and stigma when seeking treatment?  
3. If you could redesign the recovery process, what would you change or add?  
4. How do peer or online communities influence your motivation to recover?  
5. What features in an app would make you feel safe and supported?

### 2. For Professionals
1. What treatment approaches do you rely on most often for ED recovery?  
2. Which barriers do you observe most frequently in patient adherence?  
3. How do you think technology (apps, online platforms) could complement your work?  
4. What current systems or resources are most effective for your clients?  
5. What gaps exist between what professionals provide and what patients seek?

---

## Next Steps (Phase 2 Prep)
- Conduct follow-up interviews using the guide above.  
- Extract new “wish” and “unmet need” themes from student responses.  
- Add *Phase 2 sections* in the dashboard:
  1. **Unmet Needs / Wishes for Support**  
  2. **Treatment Gaps (Professionals vs Individuals)**  
  3. **Qualitative Highlights (quotes / word clouds)**  
- Begin drafting design recommendations for the Mirror app prototype.
