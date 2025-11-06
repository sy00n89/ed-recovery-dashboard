# ED Recovery Dashboard — Phase 1 (Discovery & Charter)

**Owner:** Reese Yoon  
**Date:** November 06, 2025


## What’s Here (Phase 1)
- **`docs/Phase1_Project_Charter.md`** — Client-framed charter with SMART metrics, scope, risks, and governance.
- **`docs/Phase1_Discovery_Plan_and_Interview_Guide.md`** — Interview guide + discovery methods.
- **`docs/Stakeholder_Map_RACI.csv`** — Stakeholders and responsibilities.
- **`docs/Requirements_Backlog.csv`** — Initial MoSCoW-prioritized backlog.
- **`data/raw/`** — Your current survey datasets (copied here).
- **`scripts/audit_data.py`** — Quick data audit (shapes, columns, basic PII checks).

> **Rule:** Only use the current data. No external data sources will be added in Phase 1.

## Local Setup (macOS / VS Code)

1. **Install prerequisites**
   - Install **Homebrew** (if you don’t have it): https://brew.sh
   - `brew install git python`

2. **Clone or unzip this repo**
   - If you downloaded a ZIP: unzip it and `cd ed-recovery-dashboard`
   - Or initialize Git (if starting locally):  
     ```bash
     cd ed-recovery-dashboard
     git init
     ```

3. **Create a Python virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Open in VS Code**
   - Install **VS Code**: https://code.visualstudio.com/
   - Recommended extensions: **Python**, **Pylance**, **GitLens**.
   - `code .`

5. **Run the Phase 1 data audit**
   ```bash
   source .venv/bin/activate
   python scripts/audit_data.py
   ```

## Connect to GitHub (first-time)

1. Create a new **empty** GitHub repo named `ed-recovery-dashboard` (no README/.gitignore).
2. In Terminal from this folder:
   ```bash
   git add .
   git commit -m "chore: initialize Phase 1 with data + artifacts"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/ed-recovery-dashboard.git
   git push -u origin main
   ```

> If Git asks for credentials, log into GitHub and use **Personal Access Token (classic)** with `repo` scope as your password, or use GitHub Desktop.

## Next (Phase 2 preview)
- Synthesize research → personas + journey maps
- Define experience principles and KPIs
- Prepare data feature list for the dashboard MVP

