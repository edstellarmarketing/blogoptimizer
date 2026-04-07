# GSC Cluster Engine — Streamlit App

Upload a Google Search Console export, cluster queries semantically, find content gaps against an existing blog, and generate a search-driven blog outline.

Built for country-by-country analysis of the Edstellar "In-Demand Skills in [Country]" blog series.

## Features

- **Semantic clustering** — groups queries by meaning, not just keywords (via sentence-transformers + HDBSCAN)
- **Auto-labeled clusters** — TF-IDF-based topic naming for each cluster
- **Opportunity scoring** — ranks clusters by search demand × ranking gap
- **Gap analysis** — detects which search clusters aren't covered by an existing blog
- **Blog outline generator** — produces H2/H3 structure with writer notes
- **Interactive 2D cluster map** — visual exploration of the semantic space
- **Multi-sheet Excel export** — cluster summary, keyword assignments, content briefs, blog outline, gap analysis
- **Markdown export** — copy-ready outline for writers

## Setup

### Local

```bash
# Clone or download app.py and requirements.txt into a folder
cd gsc-cluster-engine

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

First run will download the embedding model (~90 MB for MiniLM), which takes a minute.

### Streamlit Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo and deploy
4. No additional config needed

## How to use

1. **Sidebar** — set the country name, target year, clustering parameters, and existing blog URL
2. **Upload** — drop in a GSC export (xlsx / csv). Expected columns: `Top queries`, `Clicks`, `Impressions`, `CTR`, `Position`
3. **Existing blog H2s** — paste the current H2 headings of the blog you want to optimize (defaults preloaded for the Australia blog)
4. **Run** — click the big button
5. **Explore** — five tabs cover cluster map, cluster summary, gap analysis, blog outline, and export

## Running for a new country

1. Change the **Country** field in the sidebar
2. Update the **H2 sections** textarea if the target blog has a different structure
3. Upload the new GSC file
4. Click Run

Each run generates a country-specific Excel report (e.g., `gsc_clusters_canada.xlsx`).

## Tuning

| What you want | What to change |
|---|---|
| More granular clusters | Lower `Min cluster size` to 3 |
| Fewer, broader clusters | Raise to 8-10 |
| More queries assigned | Lower `Min samples` to 1 |
| Better embedding quality (slower) | Switch to `all-mpnet-base-v2` |
| Drop low-value queries | Set `Min impressions` to 10+ |
| More gap sections | Lower `Match threshold` |
| Fewer gap sections | Raise `Match threshold` to 0.45+ |

## Opportunity scoring

```
opportunity = log(total_impressions) × position_gap × ctr_penalty

Position gap:
  Pos 1-3   → 0.3x  (already winning)
  Pos 4-10  → 1.5x  (page 1 but not top 3 — best ROI)
  Pos 11-20 → 2.0x  (page 2 — high potential)
  Pos 20+   → 1.0x  (page 3+ — harder lift)

CTR penalty:
  CTR < 2%  → 1.3x boost (title/snippet problem)
```

## Blog outline color coding (Excel)

| Color | Meaning |
|---|---|
| Dark blue | H1 title |
| Yellow | 🆕 NEW section recommended from gap clusters |
| Red | Existing H2 with NO search signal (consider removing) |
| White | Validated existing section or H3 sub-section |
