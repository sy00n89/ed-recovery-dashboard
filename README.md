# Mirror — ED Recovery Insight Dashboard

*Student-run social impact startup project at Syracuse University*

**Owner**: Reese Yoon
**Last Updated**: November 2025

## Overview

Mirror is a student-led research and product initiative focused on redefining recovery systems for eating disorders (ED). It bridges clinical care and lived experience through human-centered design, data-driven insights, and iterative prototyping.

The live Streamlit dashboard integrates real-time survey data from Google Sheets to synthesize user insights and guide product hypotheses for Mirror’s future mobile app.

## Repository Structure

- app.py — Streamlit dashboard (Phases 1–2 complete; Phase 3 prototype testing in progress)
- requirements.txt — Python dependencies
- data/ — Feedback logs and live CSV storage (Phase 3)
- docs/Phase1_Project_Charter.md — Initial charter (objectives, scope, metrics)
- docs/Phase2_Project_Charter.md — Updated charter for insight synthesis and hypothesis development
- docs/Phase1_Discovery_Plan_and_Interview_Guide.md — Qualitative interview guide and discovery framework
- scripts/audit_data.py — Data quality and PII audit script

## Project Phases

**Phase 1 — Discovery & Data Collection**

- Designed and distributed two Google Forms:
   - Help Us Build a Better ED Recovery System (Individuals/Students)
   - Survey for Medical Professionals (Clinicians)
- Automated data pipeline: Google Sheets → CSV export → Streamlit auto-refresh
- Dashboard sections:
   - A) Participant Overview
   - B) Barriers to Recovery
   - C) Helpful Supports & Alignment
- Key insight: Cost, access, and stigma dominate barriers; peer connection is underrepresented in formal care models

**Phase 2 — Insight Synthesis & Hypothesis Development**
- NLP-driven keyword theming for Unmet Needs and Treatment Gaps
- Auto-promotion of “Other” responses using n-gram frequency analysis
- Qualitative Highlights for empathy mapping
- Three experience hypotheses for prototype testing:
   - Listen Mode
   - Identity Peer Spaces
   - Meal Micro-Journeys
- Reduced “Other” classification by more than 60%

**Phase 3 — Prototype & Validation (In Progress)**
- Integrated three prototypes into the live dashboard
- Collects validation feedback (sliders and text) stored in data/prototype_feedback.csv
- Real-time validation scorecard (helpfulness, empathy, reuse intent)
- Target outcomes:
   - ≥ 3.75 average “felt listened”
   - ≥ 3.5 average “helpfulness”
   - ≥ 3.0 average “reuse intent”

## Market & Research Validation
- Sources: IBISWorld, PubMed, BioMed Central, Verywell Mind, Healthy Minds Network
- Industry highlights:
   - U.S. ED clinic market size: $4.1B (2024)
   - Growth forecast: ~3.5% annually
   - Gap: Limited platforms combining peer support, identity safety, and clinical resource matching
- Evidence-based design principles:
   - Peer support reduces body dissatisfaction and anxiety
   - App-based interventions reduce binge-eating behaviors
   - Peer-guided programs improve long-term engagement
   - Behavior change techniques (goal-setting, feedback, reflection) inform design

## Local Setup (macOS/VS Code)
1. Install prereqisites
   - Install Homebrew: https://brew.sh
   - brew install git python
2. Clone this repo
```bash
git clone https://github.com/<YOUR_USERNAME>/mirror-insight-dashboard.git
cd mirror-insight-dashboard
```
3. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
4. Run Streamlit 
```bash
streamlit run app.py
```

## Deployment

**Option 1 - Streamlit Cloud**
1. Push the repo to Github 
2. Deploy at share.streamlit.io
3. Select app.py and publish

**Option 2 - Temporary ngrok Tunnel**
```bash
streamlit run app.py
ngrok http 8501
```

## Tech Stack 
- Frontend: Streamlit, Plotly Express
- Data Processing: Pandas, NumPy, Regex NLP
- Live Data Source: Google Sheets → CSV Export
- Storage (Phase 3): Local CSV feedback logger
- Environment: Python 3.12, macOS, VS Code

## Key Learnings 
- Cost, stigma, and limited identity-specific support remain major barriers
- Individuals value empathy and peer connection more than formal therapy structures
- Professionals emphasize structured modalities (CBT, DBT, FBT)
- Data → Insight → Prototype loop shortened to under one week per phase
- The dashboard demonstrates a human-centered research path to actionable, measurable design hypotheses

## Roadmap
- Phase 1 — Discovery & Charter: Complete
- Phase 2 — Insight Synthesis & Hypotheses: Complete
- Phase 3 — Prototype & Validation: In Progress
- Phase 4 — MVP Design Specification: Planned

## About Mirror's Outreach/Research Team (Startup Summary)
- Vision: Make recovery accessible, data-informed, and community-centered
- Differentiator: Bridges lived experience and professional care using live analytics and AI-driven synthesis
- Approach: Research → Insight → Prototype → Validate

## Author
- Reese (Siwoo) Yoon
- Information Management & Technology, Syracuse University
- Student Researcher, Data Analyst, Cloud and UX Enthusiast
