# Mirror — Insight Dashboard
### Project Charter (Phase 1 Completed)

**Purpose**  
Mirror aims to redefine recovery systems for eating disorders (ED) by integrating feedback from both individuals who have experienced ED and professionals who provide care.  
The project’s goal is to build a digital platform (partnered with universities) that facilitates more effective recovery and community support, guided by data-driven insights.

**Objectives**
- Collect and analyze survey data from two populations:  
  1. **Individuals (students)** — lived experiences and desired app features  
  2. **Professionals** — treatment approaches, barriers, and recommended supports  
- Translate qualitative feedback into measurable insight categories (barriers, supports, unmet needs).  
- Create a clear, automated dashboard that visualizes insights in real time.

**Scope**
- **In Scope:** Survey design, data cleaning, auto-updating dashboard development, Phase 1 analysis (overview, barriers, supports).  
- **Out of Scope (for now):** Full product prototyping, backend integration, predictive modeling.

**Deliverables (Phase 1)**
- Google Forms for students and professionals.  
- Cleaned datasets stored in linked Google Sheets.  
- Live Streamlit dashboard (`app.py`) connected via public CSV exports:
  - *Help Us Build a Better ED Recovery System (Students)*
  - *Survey for Medical Professionals*
- Automated visualizations for:
  - Respondent overview (demographics)
  - Barriers to recovery (combined)
  - Helpful supports & alignment (individuals vs. professionals)

**Key Achievements**
- Built a **fully live Streamlit dashboard** that syncs from Google Sheets every 60 s.  
- Implemented **automatic column inference**, so no manual mapping is required.  
- Transformed qualitative data into **quantitative insights** using keyword-based roll-ups.  
- Established a visual framework for future phases (Unmet Needs, Treatment Gaps, Qualitative Insights).

**Stakeholders**
- **Primary Analyst:** Reese Yoon  
- **Advisors / Collaborators:** University partners, ED professionals, and student participants  
- **End Users:** Universities offering ED recovery programs, clinicians, app design teams

**Next Phase (Phase 2 Preview)**
- Deep-dive analysis on *Wishes / Unmet Needs* and *Treatment Gaps*.  
- Add qualitative highlight cards and sentiment-based summaries.  
- Produce actionable design recommendations for app structure and content.

