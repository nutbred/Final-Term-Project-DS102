# Research History: AromaLatent

> [!IMPORTANT]
> **important: updating research_history when making file edit**


This document tracks our attempts, failures, and successes in the AromaLatent project to avoid redundant work and "failure loops."

## Current Status
- [x] Integrate `undetected-chromedriver` and `selenium`
- [x] Update SQLite schema for new data fields
- [x] Implement robust XPath-based extraction
- [x] Add review infinite scroll support
- [x] Research ML-only implementation direction for dupe detection, latent clustering, and note co-occurrence graphs
- [x] Scaffold ML MVP scripts and generate first feature, clustering, and note-graph artifacts
- [x] Generate ML plots and Vietnamese LaTeX report bundle for DS102.Q21
- [x] Generate Beamer slides explaining corrected data/ML architecture
- [x] Revise Beamer slides to Vietnamese-first copy, fix Vietnamese font setup, simplify the data flowchart, and explain why full-catalog ML can coexist with Reddit subset enrichment
- [x] Create a 6-minute 11-slide Beamer version that keeps the full story but compresses details for presentation timing
- [x] Create final 10-slide Beamer deck scoped to the 636 Reddit-crawled perfumes, with regenerated similarity and PCA/clustering figures

## Attempts & Failures

| Date | Attempt | Result | Notes |
| :--- | :--- | :--- | :--- |
| 2026-04-14 | Initial Workspace Audit | Success | Found `proposal.md` and `PerfumAPI` in `github_project_references`. |
| 2026-04-14 | PerfumAPI Analysis | In Progress | Identified dependencies: FastAPI, BeautifulSoup4, Supabase. Requires Supabase credentials. |
| 2026-04-14 | PerfumAPI Dry Run | Failure | UnicodeEncodeError on Windows due to emojis in console logs (`\U0001f9ea`). |
| 2026-04-14 | Connectivity Test | Failure | 429 Too Many Requests. Fragrantica is protected by Cloudflare JS challenges. `requests` is blocked. |
| 2026-04-15 | Selenium Migration | Success | Migrated to `undetected-chromedriver`. Successfully bypassed Cloudflare in initial tests. |
| 2026-04-15 | Schema Upgrade | Success | Added `accords`, `dupes`, `reviews` columns to perfumes table. |
| 2026-04-15 | XPath Extraction | Success | Implemented robust XPath for Main Accords, Pyramid, Dupes, and Reviews. |
| 2026-06-02 | ML Track Research Pass | Success | Added `ml_track_research_plan.md` with paper-backed implementation plan for feature engineering, dupe detection, clustering, and note graph analysis. |
| 2026-06-02 | ML Track MVP Scaffold | Success | Added `ml_track/` scripts for weighted feature extraction, descriptor-space dupe candidates, clustering, and note co-occurrence graph analysis. |
| 2026-06-02 | Vietnamese ML Report | Success | Added report figures, full Vietnamese LaTeX report source, and `report/aromalatent_ml_latex_bundle.zip`; local LaTeX compilation unavailable. |
| 2026-06-04 | Data/ML Beamer Slides | Success | Added `report/slides/aromalatent_data_ml_slides.tex` and a portable zip bundle with figures/logo for Overleaf. |
| 2026-06-04 | Data/ML Beamer Slide Revision | Success | Rewrote the deck in Vietnamese, kept technical terms where useful, added XeLaTeX/polyglossia setup for Vietnamese, replaced the overlapping flowchart, and added the full dataset vs Reddit subset explanation. |
| 2026-06-04 | 6-Minute Beamer Deck | Success | Added `report/slides/aromalatent_data_ml_slides_6min.tex` and `report/slides/aromalatent_data_ml_slides_6min_bundle.zip`; 11 slides covering architecture, Reddit crawl rationale/mechanics, ML route, results, and next steps. |
| 2026-06-04 | Reddit-Scope 10-Slide Deck | Success | Added `report/slides/perfume_similarity_10slides.tex`, `report/slides/perfume_similarity_10slides_bundle.zip`, and `report/scripts/generate_reddit_scope_figures.py`; all slide ML claims are scoped to 636 perfumes with Reddit reviews. |
| 2026-06-04 | 10-Slide Deck Cleanup | Success | Removed the overlapping label in the data architecture flowchart and added a short Creed Aventus DeepSeek-profile example to the vector slide. |
| 2026-06-04 | 10-Slide Deck Example Pass | Success | Updated slide examples with raw English-style Aventus profile output and added highest/lowest similarity pairs to the similarity slide. |
| 2026-06-04 | 10-Slide Deck Cluster Table | Success | Updated slide 4 wording to mention the Kaggle Fragrantica dataset as the initial catalog for Reddit lookup and added an intra-cluster fresh/summer similarity table to slide 9. |
| 2026-06-04 | 10-Slide Deck PCA Simplification | Success | Removed the overflowing 3x3 table on slide 9, added a concise unsupervised-cluster interpretation, and removed numeric scores from slide 8 examples. |
| 2026-06-04 | 10-Slide Deck PCA Table Layout | Success | Reworked slide 9 into two columns with the PCA figure and two bullets on the left, plus a compact 4x4 intra-cluster similarity table for the fresh/summer cluster on the right. |
| 2026-06-04 | 10-Slide Deck Final Layout Tweaks | Success | Reworked slide 8 into a two-column similarity/example layout with `EDT_vs_Parfum.png`, and replaced the conclusion follow-up block with a large thank-you line. |
| 2026-06-04 | 10-Slide Deck Formula Pass | Success | Simplified the Reddit crawler wording for a casual audience and added the pre-normalization RRF similarity formula to slide 8. |
| 2026-06-04 | 10-Slide Deck Final Cleanup | Success | Removed the RRF formula block from slide 8 after layout review and rebuilt the final Overleaf bundle. |
| 2026-06-08 | Submission Package | Success | Created `submission/` with the main Vietnamese LaTeX report, final 10-slide deck, source code, SQLite database, and separated Fragrantica catalog vs Reddit crawled review exports. |

## Cautions & Observations
- **Current source split**: The production data route should treat `fragrantica_dataset/fra_cleaned.csv` as the static source-of-truth catalog. Active crawling is targeted Reddit enrichment for selected perfume seeds, not Fragrantica review crawling.
- **Fragrantica Scraping**: Fragrantica is known for aggressive anti-scraping measures (Cloudflare and JavaScript-rendered review content). Do not make direct Fragrantica review crawling a dependency for the current DS102 delivery.
- **Data Storage**: PerfumAPI defaults to Supabase (PostgreSQL). For our DS/NLP project, we might want something more local (SQLite/Parquet) for fast iteration, or adhere to the Supabase plan if scalability is needed.
- **Dependencies**: PerfumAPI requires Python 3.9+.

## Next Steps
1. Compile `report/slides/perfume_similarity_10slides.tex` with XeLaTeX on Overleaf or another TeX environment and visually check Vietnamese glyphs.
2. Evaluate ML dupe results against the known chains in `pipeline_v3.py`.
3. Improve cluster naming by ranking overrepresented notes/accords per cluster, not only centroid terms.
4. Install optional ML packages if stronger clustering is needed: `umap-learn` and `hdbscan`.
5. Add a concise ML report section to the final project write-up using `output/ml/cluster_report.md` and `output/ml/note_graph_report.md`.

# FRAGRANTICA SCRAPING NUANCES & CONSTRAINTS

When writing the extraction logic for Fragrantica pages, you must account for the following DOM behaviors:

1. **Locators:** Use relative text-anchor XPath traversals. Do not use brittle, absolute class names (e.g., `div[class^='Grid']`) as they are dynamically generated and change frequently. Find the section header text and navigate to the nearest data container.
2. **Main Accords:** Extracted from the top visual bar chart. Capture both the accord name (e.g., 'woody') and its relative width/style percentage to determine its dominance.
3. **Dupe Network (Reminds me of):** Extract the target perfume URL/Name AND the inner text of both the 'thumbs up' and 'thumbs down' vote counters. 
4. **Note Pyramid:** Account for both traditional 3-tier structures (Top/Mid/Base) and single-tier (Linear) note lists. Extract the name of the note and, if available, the image `alt` text mapping to it.
5. **Review Infinite Scroll:** Assume reviews require Javascript execution to load. The DOM must be scrolled to `document.body.scrollHeight` multiple times, waiting for network idle, before passing the `page_source` to BeautifulSoup.
