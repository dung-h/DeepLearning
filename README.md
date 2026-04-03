# CO3133 Project Workspace

This repo is organized by assignment so research assets, notebooks, and reports do not bleed across phases.

## Main Structure

- `index.html`, `assignments/`, `assets/`: shared course landing page
- `btl1/`: notebooks, data, artifacts, reports, and demo assets for Assignment 1
- `btl2/`: placeholder space for Assignment 2
- `btl3/`: placeholder space for Assignment 3
- `docs/`: shared course documents and specs
- `legacy/`: older exploratory work that is no longer part of the main pipeline
- `logs/`: local execution logs
- `reports/`: compatibility wrappers that redirect to the real pages in `btl1/reports/`
- `AGENTS.md`: stable repo-wide working rules for Codex
- `GPT.md`: short current-state snapshot for long-running work
- `tracking_progress.txt`: lightweight repo-level progress log

## Current Status

- `BTL1` is the only assignment with active implementation.
- `BTL2` and `BTL3` are not active pipelines yet.
- Current primary notebooks in `BTL1`:
  - `btl1/notebooks/text_classification.ipynb`
  - `btl1/notebooks/text_image_classification.ipynb`
  - `btl1/notebooks/image_classification.ipynb`

## Repo Hygiene

- Keep raw datasets, archives, and generated artifacts out of the repo root.
- Keep assignment-specific executable work inside its assignment folder.
- Prefer redirect wrappers in `reports/` instead of duplicating report pages.
- Treat `legacy/` as historical reference only, not as part of the active pipeline.
