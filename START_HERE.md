# NASA Space Apps 2026 — preparation only

**Prepared on:** 15 October 2026  
**Team name under discussion:** NeuraSilicon Space AI  
**Event:** NASA Space Apps Challenge, 14–15 November 2026  
**Local event:** Hyderabad, Pakistan (virtual)  
**Challenge selected for planning:** Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data

## Rule to preserve eligibility

This folder contains training software and planning notes, **not the solution for submission**. Official 2026 Space Apps event organizers say participants should not develop their challenge project until the hackathon begins on November 14. The full official challenge statement is scheduled for October 28. Read the official current rules and consult your local event lead if needed.

Do not submit, relabel, or treat the pre-built practice dashboard as competition work. Begin a NEW solution repository during the competition itself.

Official event page: https://www.spaceappschallenge.org/2026/  
Official challenge page: https://www.spaceappschallenge.org/2026/challenges/flame-in-freefall-ai-powered-fire-safety-insights-from-microgravity-combustion-data/  
Local event example of explicit rule: https://www.spaceappschallenge.org/2026/local-events/santiago-dominican-republic/

## Today: technical skills preparation

The included Streamlit dashboard uses **synthetic electronics-lab sensor measurements**. It is intentionally unrelated to NASA's challenge and teaches general-purpose CSV ingestion, checking, selection, summaries, and charting. It is NOT NASA data, a fire-safety model, or an AI product.

### Windows PowerShell steps

```powershell
cd NASA_Space_Apps_2026_Preparation
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python create_sample_data.py
python -m unittest discover -s tests -v
python -m streamlit run practice_dashboard.py
```

If PowerShell prevents activation, run `cmd` and then `.venv\Scripts\activate.bat` instead.

Then visit the localhost URL displayed in the terminal (typically http://localhost:8501).

### Practice tasks

1. Explain what `read_records()` checks and why missing values should not be silently ignored.
2. Filter the example data to Bench-A and confirm the metrics change.
3. Upload a malformed CSV and see its row-level error.
4. Compare time-series charts for the two training benches.
5. Run the unit tests and confirm all pass.

## Preparation research (no competition implementation yet)

Read the official 2026 challenge summary and prepare a list of questions to resolve once the full instructions arrive October 28. Review, but do not implement project data pipelines for:

- NASA Physical Sciences Informatics (PSI): https://psi.nasa.gov/physci/repo
- FLEX microgravity investigation (PSI-69): https://psi.nasa.gov/physci/repo/data/investigations/PSI-69
- BASS-II microgravity investigation (PSI-25): https://psi.nasa.gov/physci/repo/data/investigations/PSI-25
- NASA combustion-science background: https://www.nasa.gov/missions/station/iss-research/studying-combustion-and-fire-safety/

FLEX and BASS-II are real NASA investigations. Individual files will require inspection for exact format, sampling, quality, units, licensing, citations, and suitability; their landing pages alone do NOT validate a predictive model.

## High-level weekend plan — start actual work on November 14

- Confirm official brief and permissible data files; agree on user stories and measures of success.
- Start a brand-new project repository **during** the hackathon.
- Import selected real data and preserve provenance (investigation ID, DOI, source URL, data license, units).
- Implement exploratory charts, evidence-backed comparisons and transparent limitations.
- Add an AI interpretation component only if its responses can be grounded in identified NASA source material; no fabricated findings or unvalidated spacecraft safety prescriptions.
- Test edge cases, document reproducibility, prepare short walkthrough, submit through your team page before the local deadline.

## Research ethics and scientific validity

- Never label synthetic examples as NASA data.
- Separate measured observations, analysis, and hypotheses.
- Do not claim a practical spacecraft safety recommendation is validated without scientific evidence.
- Clearly cite NASA investigation DOI(s) and acknowledge uncertainty.
