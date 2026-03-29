---
title: Web Auditor Target
emoji: 💻
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# OpenEnv: Web Accessibility & SEO Auditor

## Motivation and Description
This environment fundamentally acts as a pragmatic simulation of a "Webmaster" auditing a beautifully designed but severely broken static marketing website (The "Travel Journal"). This strictly complies with the Evaluation Criteria mandate: **"Must simulate a real-world task (not games or toys)"**. Finding broken DOM elements, navigating HTML structural sabotage, and fixing strict metadata are vital daily tasks for modern web operation engineers.

## Spaces
Fully compliant with official `openenv-core` standards:
- **Action Space**: `WebAuditorAction` containing a `command: str` parameter. The agent uses standard Bash inputs to list directories, parse HTML (`cat`), and apply edits (`python`, `sed`, etc.).
- **Observation Space**: `WebAuditorObservation` returns continuous contextual state on each tick.
  - `output` (The raw stdout from your executed bash tool)
  - `current_directory_structure` (The immediate `ls -R` visual map of the root folders)
  - `file_content` (The fully rendered text structure of an investigated file, streamlining agent ingestion)

## Task Graders
Grader scripts process deterministic performance scores ranging from `0.0` to `1.0` dynamically combining into a unified reward system:

1. **Easy Task (Image Accessibility):** Searches the DOM for `<img alt="...">`. Validates meaningful accessibility strings over 3 characters and divides by total images.
2. **Medium Task (Heading Strictness):** Extracts `<h>` tags maintaining linear HTML5 semantic sequential order. Penalizes skipped hierarchies (e.g., `H1 -> H4`).
3. **Hard Task (Sitemap generation):** Validates the `sitemap.xml` mapping schema to ensure all paths dynamically resolve.

## Setup & Deployment Instructions
1. Install dependencies: `pip install -r server/requirements.txt`
2. Validate compliance: `openenv validate .`
3. Spin up the test endpoint server: `uv run server` (Hosts the `FastAPI` endpoint locally on `localhost:7860`)
4. Ping your space via standard HTTP `POST /reset` to retrieve initial observation blocks!

## Baseline Scoring
Running the internal `inference.py` script returns a reproducible Baseline starting score of `0.00` because the native website template is purposefully injected with missing `alt` attributes, skipped `<h>` tags, and missing structural sitemaps! Once the `OpenAI` LLM natively executes remediation scripts against the working directory files, the `/step` loop cleanly emits the partial progress metrics mapped out above.
