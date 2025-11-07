# Mirror — Insight Dashboard
### Project Charter (Phase 2 Updated)

**Purpose**  
Mirror aims to redefine recovery systems for eating disorders (ED) by integrating feedback from both individuals who have experienced ED and professionals who provide care.  
The project’s goal is to build a digital platform (partnered with universities) that facilitates more effective recovery and community support, guided by data-driven insights.

---

<details>
<summary><strong>▼ Phase 2 (Insight Synthesis & Experience Hypothesis Development)</strong></summary>

**Purpose**
Leverage human-centered, data-driven analysis to bridge the gap between clinical care and lived experience through a scalable digital recovery support ecosystem.

**Objectives**
- Transform open-text survey data into quantified experience insights and actionable opportunity areas.
- Deliver a live, stakeholder-ready dashboard for continuous insight review and design decisions.
- Align student and clinician perspectives to inform co-designed features that enhance engagement and recovery outcomes.

**Scope**
- Theming and auto-promotion of “wishes” and unmet needs using n-gram analysis.
- Quantitative comparison of individual vs. professional perspectives (treatment gaps).
- Curated qualitative quotes for empathy mapping and design alignment.
- Three evidence-backed experience hypotheses with early adoption metrics.

**Deliverables**
1. Enhanced Streamlit dashboard with new Sections D–F (Wishes, Treatment Gaps, Qualitative Highlights).
2. Gap-analysis visualizations and delta table comparing student and professional priorities.
3. Quote summary cards for design and research documentation.
4. Experience hypothesis portfolio: “Listen Mode,” “Identity-Based Peer Spaces,” “Meal Micro-Journeys.”

**Key Achievements**
- Reduced “Other” category by over 60 % through automated n-gram promotion and theme generation.
- Mapped three student-defined themes to clinician equivalents, revealing specific experience gaps.
- Delivered a real-time, automatically refreshing dashboard sourced from public Google Sheets.
- Produced clear, data-grounded hypotheses framed as both business and user outcomes—ready for prototype validation.

**Success Metrics (Phase 2)**
- ≥ 80 % of qualitative responses classified into defined themes (non-Other).
- 3–6 high-confidence gap areas with representative quotes.
- Stakeholder alignment and approval of the top 3 hypotheses for Phase 3 prototyping.

**Next Phase**
- Prototype and test the three hypotheses with real users.
- Measure perceived helpfulness and intent-to-reuse over two weeks.
- Translate validated concepts into an MVP specification

</details>

<details>
<summary><strong>▼ Phase 1 (Background / Initial Discovery)</strong></summary>

**Objectives**
- Collect and analyze survey data from two populations:  
  1. **Individuals (students)** — lived experiences and desired app features  
  2. **Professionals** — treatment approaches, barriers, and recommended supports  
- Translate qualitative feedback into measurable insight categories (barriers, supports, unmet needs).  
- Create an automated dashboard that visualizes insights in real time.

**Scope**
- **In Scope:** Survey design, data cleaning, live dashboard development, and baseline analysis (Overview, Barriers, Supports). 
- **Out of Scope (for now):** Full product prototyping, backend integration, predictive modeling.

**Deliverables (Phase 1)**
- Two Google Forms (students and professionals). 
- Cleaned datasets automatically synced to public Google Sheets.
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
- Conduct deeper analysis of Wishes / Unmet Needs and Treatment Gaps.
- Add qualitative highlight cards and sentiment summaries. 
- Develop experience hypotheses to guide MVP feature design.

</details>

