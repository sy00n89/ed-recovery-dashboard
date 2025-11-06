# Phase 1 — Project Charter  
**Project:** Mirror — Live Insight Dashboard for Eating-Disorder Recovery Research  
**Prepared by:** Siwoo Yoon (Outreach & Insights, Mirror Startup)  
**Date:** Nov 6 2025

---

## 💡 1. Background & Problem Statement
Universities and colleges face persistent gaps in accessible, inclusive eating-disorder support.  
Mirror, a **student-led startup**, is building a **free, peer-led, evidence-based recovery app** that campuses can license and offer to their students.

To guide Mirror’s product design, the Outreach & Insights team is developing a **real-time, cloud-based insight dashboard** that visualizes survey results from students and medical professionals collected through **Google Forms**.  
The dashboard will continuously pull new responses, summarize trends, and surface actionable findings for the product and UX teams.

---

## 2. Objectives
- **O1 – Centralize Data:** Connect Google Forms responses to a live dashboard that updates automatically as new entries arrive.  
- **O2 – Visualize Insights:** Build interactive charts showing trends in barriers, help-seeking behavior, and desired features.  
- **O3 – Guide Product Decisions:** Translate data insights into clear recommendations for Mirror’s UX and development teams.  
- **O4 – Maintain Data Ethics:** Handle all survey data securely and exclude any personally identifiable information (PII).

---

## 3. Success Metrics (SMART)
| Metric | Target |
|--|--|
| Google Forms integration functional | Live sync via Google Sheets API or direct CSV import every 24 h |
| Dashboard visualizations | ≥ 5 interactive charts (filterable by role and topic) |
| Insight accuracy | 100% match to Google Form data |
| Stakeholder usability rating | ≥ 4.5 / 5 from Mirror leadership team |
| Security | No PII stored locally or publicly shared |

---

## 4. Scope
**In Scope (Phase 1):**  
- Audit and clean existing Google Form data (CSV exports).  
- Set up a local data pipeline from Google Sheets API → dashboard app (Streamlit or Flask + Plotly).  
- Create visual summaries of key themes (barriers, resources, desired features).  
- Document technical setup for future automation.

**Out of Scope (Phase 1):**  
- App UI/UX design and clinical content creation.  
- User authentication or role-based dashboards (those come Phase 3).

---

## 👥 5. Stakeholders
| Category | Stakeholder | Role |
|--|--|--|
| Internal | Haley | Approves dashboard scope and timeline |
| Internal | Siwoo Yoon (Outreach) | Leads data integration, dashboard build, and insight presentation |


---

## 6. Risks & Mitigation
| Risk | Mitigation |
|--|--|
| Google Sheets API limits | Use manual CSV backup sync and API quota monitoring |
| Data privacy concerns | Strip emails and personal fields before loading |
| Dashboard performance | Paginate large datasets and cache results |
| Interpretation bias | Review findings with team and clinician advisors |

---

## 7. Deliverables
- Google Forms → Sheets → Dashboard data pipeline (setup instructions documented).  
- Interactive dashboard (Streamlit or Flask) with live charts and filters.  
- Data dictionary and column mapping sheet.  
- Brief readout deck summarizing insights for leadership review.  
- Phase 1 GitHub repo with clean and version-controlled code.

---

## 8. Governance & Ethics
- No email or name fields will be used in any analysis or stored locally.  
- All scripts and data reside in a private GitHub repository.  
- Outputs represent aggregate trends only — no individual diagnostic interpretation.

---

## 9. Phase 1 → Phase 2 Handoff
**Output of Phase 1:** Live dashboard + documentation.  
**Next Step:** Use dashboard findings to build user personas and journey maps for the Mirror prototype.

---
