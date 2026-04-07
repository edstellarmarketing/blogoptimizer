"""
GSC Search Terms → Content Cluster Engine (Streamlit)
======================================================
Upload a Google Search Console export, cluster queries semantically,
find content gaps vs an existing blog, and generate an optimized outline.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import os
import re
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GSC Cluster Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
    h1 { color: #1e3a5f; }
    h2 { color: #1e3a5f; border-bottom: 2px solid #e8f0fe; padding-bottom: 0.3rem; }
    .stMetric { background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1e3a5f; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f5f9; padding: 8px 16px; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #1e3a5f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.title("🔍 GSC Cluster Engine")
st.caption(
    "Upload a Google Search Console export → cluster queries by meaning → "
    "find content gaps vs an existing blog → generate a search-driven outline."
)


# ============================================================
# SIDEBAR — CONFIG
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    country = st.text_input("Country / Market", value="Australia")
    target_year = st.number_input("Target Year", min_value=2024, max_value=2030, value=2026)

    st.divider()
    st.subheader("Clustering")
    min_cluster_size = st.slider(
        "Min cluster size", min_value=2, max_value=20, value=4,
        help="Lower = more granular clusters. Higher = broader themes."
    )
    min_samples = st.slider(
        "Min samples", min_value=1, max_value=10, value=2,
        help="Density threshold. Lower = fewer queries classified as noise."
    )
    umap_n_neighbors = st.slider(
        "UMAP neighbors", min_value=5, max_value=50, value=15, step=5,
        help="Lower = local structure. Higher = global context."
    )

    st.divider()
    st.subheader("Filters")
    min_impressions = st.number_input(
        "Min impressions filter", min_value=0, value=0,
        help="Drop queries below this impression count"
    )

    st.divider()
    st.subheader("Embedding Model")
    embedding_model_name = st.selectbox(
        "Model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
        help="MiniLM = fast. MPNet = better quality but slower."
    )

    st.divider()
    st.subheader("Blog Gap Analysis")
    match_threshold = st.slider(
        "Match threshold", min_value=0.20, max_value=0.60, value=0.35, step=0.05,
        help="Cosine similarity threshold. Higher = stricter matching (more gaps found)."
    )
    blog_url = st.text_input(
        "Existing blog URL",
        value="https://www.edstellar.com/blog/skills-in-demand-in-australia"
    )
    title_template = st.text_input(
        "Title template",
        value="Top In-Demand Skills in {country} for {year}",
        help="Use {country} and {year} placeholders"
    )


# ============================================================
# CACHED MODEL LOADER
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """Load sentence-transformer model (cached across runs)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Fuzzy-match a column name from a list of possible names."""
    cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        for col_key, col_original in cols_lower.items():
            if key in col_key or col_key in key:
                return col_original
    return None


def parse_gsc_file(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded GSC export into a raw dataframe."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    elif ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_gsc(df_raw: pd.DataFrame, min_impressions: int, country: str) -> pd.DataFrame:
    """Detect and normalize GSC columns."""
    query_col = find_col(df_raw, [
        "top queries", "query", "queries", "keyword", "search_term",
        "search term", "top_queries", "keyphrase"
    ])
    clicks_col = find_col(df_raw, ["clicks", "click"])
    impressions_col = find_col(df_raw, ["impressions", "impression", "impr"])
    ctr_col = find_col(df_raw, ["ctr", "click_through_rate", "click through rate"])
    position_col = find_col(df_raw, ["position", "avg_position", "average position", "rank"])

    if query_col is None:
        # Fallback to first text column
        for c in df_raw.columns:
            if df_raw[c].dtype == "object":
                query_col = c
                break
    if query_col is None:
        raise ValueError("Cannot detect a query column. Rename it to 'Top queries' or 'query'.")

    df = pd.DataFrame()
    df["query"] = df_raw[query_col].astype(str).str.strip().str.lower()

    col_map = {
        "clicks": clicks_col,
        "impressions": impressions_col,
        "ctr": ctr_col,
        "position": position_col,
    }
    detected = {"query": query_col}
    for metric, col in col_map.items():
        if col:
            series = df_raw[col].copy()
            if metric == "ctr" and series.dtype == "object":
                series = series.astype(str).str.replace("%", "", regex=False)
                series = pd.to_numeric(series, errors="coerce") / 100
            else:
                series = pd.to_numeric(series, errors="coerce")
            df[metric] = series.fillna(0)
            detected[metric] = col

    df = df[df["query"].str.len() > 1].drop_duplicates(subset="query").reset_index(drop=True)

    if min_impressions > 0 and "impressions" in df.columns:
        df = df[df["impressions"] >= min_impressions].reset_index(drop=True)

    df["country"] = country
    return df, detected


def cluster_queries(df: pd.DataFrame, embeddings: np.ndarray,
                    min_cluster_size: int, min_samples: int,
                    umap_n_neighbors: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Run UMAP + HDBSCAN clustering and return df + 2D coords for viz."""
    import hdbscan
    import umap

    n_components_cluster = min(10, max(2, len(df) - 2))
    umap_cluster = umap.UMAP(
        n_neighbors=min(umap_n_neighbors, len(df) - 1),
        n_components=n_components_cluster,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embeddings_reduced = umap_cluster.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    df = df.copy()
    df["cluster_id"] = clusterer.fit_predict(embeddings_reduced)

    # 2D projection for visualization
    umap_2d = umap.UMAP(
        n_neighbors=min(umap_n_neighbors, len(df) - 1),
        n_components=2,
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    coords_2d = umap_2d.fit_transform(embeddings)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    return df


def label_cluster(keywords: list, country: str, top_n: int = 3) -> str:
    """Generate a human-readable topic label using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(keywords) < 2:
        return keywords[0].title() if keywords else "Misc"

    custom_stop_words = list({
        country.lower(), "australian", "aus",
        "list", "demand", "skill", "skills", "in", "of", "for", "the", "to",
        "and", "is", "are", "what", "which", "how", "best"
    })

    try:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=60,
            stop_words=custom_stop_words,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        )
        matrix = tfidf.fit_transform(keywords)
        features = tfidf.get_feature_names_out()
        scores = matrix.sum(axis=0).A1

        scored = sorted(zip(features, scores), key=lambda x: -x[1])
        # Boost multi-word terms
        for i, (term, score) in enumerate(scored):
            if " " in term:
                scored[i] = (term, score * 1.3)
        scored.sort(key=lambda x: -x[1])

        picked = []
        for term, _ in scored:
            if len(picked) >= top_n:
                break
            if not any(term in p or p in term for p in picked):
                picked.append(term)

        return " / ".join(picked).title() if picked else "Misc"
    except Exception:
        words = " ".join(keywords).split()
        words = [w for w in words if w.lower() not in custom_stop_words]
        common = Counter(words).most_common(top_n)
        return " / ".join([w.title() for w, _ in common]) if common else "Misc"


def classify_search_intent(keywords_text: str) -> str:
    """Classify the dominant search intent of a cluster."""
    t = keywords_text.lower()
    if any(w in t for w in ["how to", "how do", "guide", "steps to", "tutorial", "way to", "process"]):
        return "How-To / Guide"
    elif any(w in t for w in ["best", "top", "vs", "compare", "comparison", "alternative", "review"]):
        return "Listicle / Comparison"
    elif any(w in t for w in ["what is", "what are", "meaning", "definition", "explain"]):
        return "Explainer / Pillar Page"
    elif any(w in t for w in ["occupation list", "shortage list", "strategic skills list", "core skills"]):
        return "Reference List / Database"
    elif any(w in t for w in ["visa", "immigration", "migrate", "emigrate", "work permit", "sponsor"]):
        return "Immigration / Visa Guide"
    elif any(w in t for w in ["salary", "pay", "cost", "fee", "price", "worth", "earning"]):
        return "Data / Stats Article"
    elif any(w in t for w in ["course", "certification", "training", "learn", "program", "degree"]):
        return "Course / Training Page"
    elif any(w in t for w in ["job", "jobs", "career", "hiring", "vacancy", "role"]):
        return "Jobs / Career Content"
    else:
        return "Blog Post / Article"


def build_cluster_summary(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """One row per cluster with metrics + opportunity score."""
    has = {col: col in df.columns for col in ["clicks", "impressions", "ctr", "position"]}
    rows = []

    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        sub = df[df["cluster_id"] == cid]
        kws_text = " ".join(sub["query"].tolist())

        row = {
            "cluster_id": cid,
            "cluster_name": sub["cluster_name"].iloc[0],
            "country": country,
            "keyword_count": len(sub),
        }

        if has["impressions"]:
            row["total_impressions"] = int(sub["impressions"].sum())
        if has["clicks"]:
            row["total_clicks"] = int(sub["clicks"].sum())
        if has["ctr"]:
            row["avg_ctr"] = round(sub["ctr"].mean(), 4)
            if has["impressions"] and sub["impressions"].sum() > 0:
                row["weighted_ctr"] = round(
                    (sub["ctr"] * sub["impressions"]).sum() / sub["impressions"].sum(), 4
                )
        if has["position"]:
            row["avg_position"] = round(sub["position"].mean(), 2)
            if has["impressions"] and sub["impressions"].sum() > 0:
                row["weighted_position"] = round(
                    (sub["position"] * sub["impressions"]).sum() / sub["impressions"].sum(), 2
                )

        # Opportunity score
        demand = np.log1p(row.get("total_impressions", len(sub)))
        gap = 1.0
        if has["position"]:
            avg_pos = row.get("weighted_position", row.get("avg_position", 5))
            if avg_pos <= 3:
                gap = 0.3
            elif avg_pos <= 10:
                gap = 1.5
            elif avg_pos <= 20:
                gap = 2.0
            else:
                gap = 1.0
        if has["ctr"]:
            w_ctr = row.get("weighted_ctr", row.get("avg_ctr", 0.05))
            if w_ctr < 0.02:
                gap *= 1.3

        row["opportunity_score"] = round(demand * gap, 2)
        row["search_intent"] = classify_search_intent(kws_text)

        if has["impressions"]:
            top_kws = sub.nlargest(5, "impressions")["query"].tolist()
            row["head_keyword"] = sub.loc[sub["impressions"].idxmax(), "query"]
        else:
            top_kws = sub["query"].head(5).tolist()
            row["head_keyword"] = sub["query"].iloc[0]
        row["top_queries"] = " | ".join(top_kws)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    summary.index += 1
    summary.index.name = "rank"
    return summary


def analyze_gaps(df: pd.DataFrame, embeddings: np.ndarray, cluster_summary: pd.DataFrame,
                 existing_h2s: list, model, threshold: float):
    """Map clusters to existing H2 sections and identify gaps."""
    from sklearn.metrics.pairwise import cosine_similarity

    h2_embeddings = model.encode(existing_h2s, show_progress_bar=False)

    cluster_centroids = {}
    cluster_keyword_map = {}
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        mask = df["cluster_id"] == cid
        indices = df[mask].index.tolist()
        centroid = embeddings[indices].mean(axis=0)
        cluster_centroids[cid] = centroid
        sort_col = "impressions" if "impressions" in df.columns else "query"
        cluster_keyword_map[cid] = df[mask].sort_values(sort_col, ascending=False)["query"].tolist()

    if not cluster_centroids:
        return [], [], {}, cluster_keyword_map

    centroid_ids = list(cluster_centroids.keys())
    centroid_matrix = np.array([cluster_centroids[cid] for cid in centroid_ids])
    sim_matrix = cosine_similarity(centroid_matrix, h2_embeddings)

    covered_clusters = []
    gap_clusters = []
    h2_to_clusters = {h2: [] for h2 in existing_h2s}

    # Build lookup from cluster_summary
    cs_lookup = cluster_summary.set_index("cluster_id").to_dict(orient="index")

    for i, cid in enumerate(centroid_ids):
        best_h2_idx = sim_matrix[i].argmax()
        best_score = sim_matrix[i][best_h2_idx]
        best_h2 = existing_h2s[best_h2_idx]

        cs_row = cs_lookup.get(cid, {})
        entry = {
            "cluster_id": cid,
            "cluster_name": cs_row.get("cluster_name", "?"),
            "keyword_count": len(cluster_keyword_map[cid]),
            "top_keywords": cluster_keyword_map[cid][:5],
            "best_h2_match": best_h2,
            "similarity": round(float(best_score), 3),
            "opportunity_score": cs_row.get("opportunity_score", 0),
            "total_impressions": int(df[df["cluster_id"] == cid]["impressions"].sum())
                                 if "impressions" in df.columns else 0,
        }

        if best_score >= threshold:
            covered_clusters.append(entry)
            h2_to_clusters[best_h2].append(entry)
        else:
            gap_clusters.append(entry)

    gap_clusters.sort(key=lambda x: -x["opportunity_score"])
    return covered_clusters, gap_clusters, h2_to_clusters, cluster_keyword_map


def build_blog_outline(country: str, year: int, existing_h2s: list,
                       h2_to_clusters: dict, gap_clusters: list,
                       cluster_keyword_map: dict, df: pd.DataFrame) -> list:
    """Build a search-driven blog outline."""
    outline = []

    # H1
    outline.append({
        "level": "H1",
        "heading": f"Top In-Demand Skills in {country} for {year}",
        "type": "title",
        "keywords": [],
        "notes": f"Primary keyword: skills in demand in {country.lower()}",
        "impressions": 0,
    })

    # Intro
    intro_kws = df[
        df["query"].str.contains(
            r"skills?.*(in demand|demand|needed|required|shortage)",
            regex=True, case=False, na=False
        )
        & ~df["query"].str.contains(
            r"(visa|immigration|migrate|occupation list|482|189|190)",
            regex=True, case=False, na=False
        )
    ]["query"].head(10).tolist()

    outline.append({
        "level": "H2",
        "heading": f"Why {country} Needs Skilled Professionals in {year}",
        "type": "intro",
        "keywords": intro_kws[:5],
        "notes": "Set context: labor market overview, job creation stats, fourth industrial revolution impact. Link to ACS / Jobs and Skills Australia data.",
        "impressions": int(df[df["query"].isin(intro_kws)]["impressions"].sum())
                       if "impressions" in df.columns else 0,
    })

    # Existing H2s (enriched)
    for h2 in existing_h2s:
        matched = h2_to_clusters.get(h2, [])
        if not matched:
            outline.append({
                "level": "H2",
                "heading": h2,
                "type": "existing (no GSC signal)",
                "keywords": [],
                "notes": "⚠️ No matching search queries in GSC data. Consider: (a) keeping if editorially important, (b) merging into a broader section, or (c) removing if no value.",
                "impressions": 0,
            })
            continue

        all_kws = []
        total_impr = 0
        for entry in matched:
            cid = entry["cluster_id"]
            all_kws.extend(cluster_keyword_map.get(cid, []))
            total_impr += entry["total_impressions"]

        kw_with_impr = df[df["query"].isin(all_kws)].drop_duplicates("query")
        if "impressions" in kw_with_impr.columns:
            kw_with_impr = kw_with_impr.sort_values("impressions", ascending=False)
        unique_kws = kw_with_impr["query"].tolist()

        outline.append({
            "level": "H2",
            "heading": h2,
            "type": "existing (search-validated)",
            "keywords": unique_kws[:8],
            "notes": f"{len(matched)} cluster(s) matched. Total impressions: {total_impr:,}. Enrich with: why this skill matters in {country}, salary range, relevant certifications, Edstellar training link.",
            "impressions": total_impr,
        })

        # H3 sub-topics
        subtopic_patterns = {
            f"Why {h2} is in Demand in {country}": r"(demand|needed|required|shortage|why)",
            f"Key {h2} Certifications & Training": r"(certification|course|training|learn|program)",
            f"{h2} Salary & Job Outlook in {country}": r"(salary|pay|job|career|hiring|earn)",
        }
        for h3_title, pattern in subtopic_patterns.items():
            matching = [k for k in unique_kws if re.search(pattern, k, re.I)]
            if matching:
                outline.append({
                    "level": "H3",
                    "heading": h3_title,
                    "type": "sub-topic",
                    "keywords": matching[:4],
                    "notes": "",
                    "impressions": int(df[df["query"].isin(matching)]["impressions"].sum())
                                   if "impressions" in df.columns else 0,
                })

    # New H2s from gap clusters
    significant_gaps = [g for g in gap_clusters if g["total_impressions"] >= 50 or g["keyword_count"] >= 5]

    visa_gaps = [g for g in significant_gaps if any(
        w in " ".join(g["top_keywords"])
        for w in ["visa", "immigration", "immigra", "migrate", "emigrate", "sponsor", "work permit", "working holiday"]
    )]
    list_gaps = [g for g in significant_gaps if any(
        w in " ".join(g["top_keywords"])
        for w in ["occupation list", "shortage list", "skilled list", "core skills", "mltssl", "csol", "pmsol", "strategic skills list"]
    )]
    job_gaps = [g for g in significant_gaps if any(
        w in " ".join(g["top_keywords"]) for w in ["job", "jobs", "career", "hiring", "work", "employment"]
    ) and g not in visa_gaps and g not in list_gaps]
    assessment_gaps = [g for g in significant_gaps if any(
        w in " ".join(g["top_keywords"])
        for w in ["assessment", "vetassess", "skillselect", "acs", "home affairs"]
    )]
    other_gaps = [g for g in significant_gaps
                  if g not in visa_gaps and g not in list_gaps
                  and g not in job_gaps and g not in assessment_gaps]

    def add_gap_section(gaps, heading, notes_prefix):
        if not gaps:
            return
        all_kws = []
        total_impr = 0
        for g in gaps:
            all_kws.extend(g["top_keywords"])
            total_impr += g["total_impressions"]
        outline.append({
            "level": "H2",
            "heading": heading,
            "type": "NEW — from gap clusters",
            "keywords": list(dict.fromkeys(all_kws))[:8],
            "notes": f"{notes_prefix} {len(gaps)} gap cluster(s), {total_impr:,} total impressions.",
            "impressions": total_impr,
        })

    add_gap_section(
        visa_gaps,
        f"Skilled Visa Pathways for {country} ({year})",
        "🆕 HIGH-VALUE GAP. Cover: Skills in Demand visa (subclass 482), employer sponsorship, skilled migration pathways, working holiday. Link to Edstellar training as upskilling for visa eligibility."
    )
    add_gap_section(
        list_gaps,
        f"{country} Skilled Occupation List — What You Need to Know",
        "🆕 HIGH-VALUE GAP. Cover: MLTSSL, STSOL, CSOL, Core Skills Occupation List, how lists are updated, which Edstellar-relevant roles appear."
    )
    add_gap_section(
        job_gaps,
        f"High-Demand Jobs in {country} for Skilled Workers",
        "🆕 GAP. Cover: top industries hiring, roles for foreigners, salary expectations."
    )
    add_gap_section(
        assessment_gaps,
        f"Skills Assessment Process for {country}",
        "🆕 GAP. Cover: VETASSESS, ACS, SkillSelect, how to get skills assessed."
    )

    for g in other_gaps[:5]:
        outline.append({
            "level": "H2",
            "heading": g["cluster_name"].replace(" / ", ": "),
            "type": "NEW — from gap clusters",
            "keywords": g["top_keywords"][:6],
            "notes": f"🆕 GAP. {g['keyword_count']} queries, {g['total_impressions']:,} impressions, opp score: {g['opportunity_score']}.",
            "impressions": g["total_impressions"],
        })

    # Best sources / CTA
    outline.append({
        "level": "H2",
        "heading": f"Best Sources for Developing High-Demand Skills in {country}",
        "type": "standard section",
        "keywords": [
            f"how to develop skills for {country.lower()}",
            f"training programs {country.lower()}",
            f"upskilling {country.lower()}",
        ],
        "notes": "CTA section. Position Edstellar training programs as the solution. Link to relevant course pages.",
        "impressions": 0,
    })

    # FAQ
    if "impressions" in df.columns:
        question_queries = df[
            df["query"].str.match(
                r"^(what|which|how|why|when|where|is |are |do |does |can )",
                case=False, na=False
            )
        ].sort_values("impressions", ascending=False)
    else:
        question_queries = df[
            df["query"].str.match(
                r"^(what|which|how|why|when|where|is |are |do |does |can )",
                case=False, na=False
            )
        ]

    faq_questions = question_queries["query"].head(8).tolist()
    if faq_questions:
        outline.append({
            "level": "H2",
            "heading": "Frequently Asked Questions",
            "type": "FAQ (schema-ready)",
            "keywords": faq_questions,
            "notes": "Use FAQ schema markup. Each question = one H3. Answer in 2-3 sentences for featured snippet eligibility.",
            "impressions": int(question_queries.head(8)["impressions"].sum())
                           if "impressions" in df.columns else 0,
        })
        for q in faq_questions[:6]:
            outline.append({
                "level": "H3",
                "heading": q.title().rstrip("?") + "?",
                "type": "FAQ question",
                "keywords": [q],
                "notes": "",
                "impressions": int(df[df["query"] == q]["impressions"].sum())
                               if "impressions" in df.columns else 0,
            })

    # Conclusion
    outline.append({
        "level": "H2",
        "heading": "Conclusion",
        "type": "standard section",
        "keywords": [],
        "notes": f"Summarize key skills, reinforce {country} opportunity, CTA to explore Edstellar training programs.",
        "impressions": 0,
    })

    return outline


def build_excel_report(country: str, blog_title: str, df: pd.DataFrame,
                       cluster_summary: pd.DataFrame, blog_outline: list,
                       gap_clusters: list) -> bytes:
    """Build a multi-sheet Excel report as bytes."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr = wb.add_format({
            "bold": True, "bg_color": "#1e3a5f", "font_color": "#ffffff",
            "border": 1, "text_wrap": True, "valign": "vcenter", "font_size": 10
        })
        title_fmt = wb.add_format({"bold": True, "font_size": 14, "font_color": "#1e3a5f"})
        h1_fmt = wb.add_format({"bold": True, "font_size": 13, "bg_color": "#0d2137", "font_color": "#ffffff"})
        new_section_fmt = wb.add_format({"bg_color": "#fef3c7", "font_size": 10})
        gap_fmt = wb.add_format({"bg_color": "#fee2e2", "font_size": 10})

        # Sheet 1: Cluster Summary
        cs = cluster_summary.reset_index()
        cs.to_excel(writer, sheet_name="Cluster Summary", index=False, startrow=1)
        ws1 = writer.sheets["Cluster Summary"]
        ws1.write(0, 0, f"Content Clusters — {country}", title_fmt)
        for i, col in enumerate(cs.columns):
            ws1.write(1, i, col, hdr)
        ws1.set_column("A:A", 6)
        ws1.set_column("B:B", 10)
        ws1.set_column("C:C", 32)
        ws1.set_column("N:N", 60)
        ws1.autofilter(1, 0, len(cs) + 1, len(cs.columns) - 1)
        ws1.freeze_panes(2, 3)

        # Sheet 2: Keywords by Cluster
        df_export = df.drop(columns=["x", "y"], errors="ignore")
        sort_col = "impressions" if "impressions" in df.columns else "query"
        df_export = df_export.sort_values(
            ["cluster_id", sort_col], ascending=[True, False]
        ).reset_index(drop=True)
        df_export.to_excel(writer, sheet_name="Keywords by Cluster", index=False, startrow=1)
        ws2 = writer.sheets["Keywords by Cluster"]
        ws2.write(0, 0, f"All Queries — {country}", title_fmt)
        for i, col in enumerate(df_export.columns):
            ws2.write(1, i, col, hdr)
        ws2.set_column("A:A", 50)
        ws2.autofilter(1, 0, len(df_export) + 1, len(df_export.columns) - 1)
        ws2.freeze_panes(2, 1)

        # Sheet 3: Content Briefs
        briefs = []
        for rank, (_, row) in enumerate(cluster_summary.head(30).iterrows(), 1):
            sub = df[df["cluster_id"] == row["cluster_id"]]
            all_kws = sub["query"].tolist()
            head = row["head_keyword"]
            secondary = [k for k in all_kws if k != head][:8]
            briefs.append({
                "priority": rank,
                "cluster_name": row["cluster_name"],
                "search_intent": row["search_intent"],
                "primary_keyword": head,
                "secondary_keywords": " | ".join(secondary),
                "total_keywords": len(all_kws),
                "total_impressions": row.get("total_impressions", ""),
                "opportunity_score": row["opportunity_score"],
                "country": country,
            })
        df_briefs = pd.DataFrame(briefs)
        df_briefs.to_excel(writer, sheet_name="Content Briefs", index=False, startrow=1)
        ws3 = writer.sheets["Content Briefs"]
        ws3.write(0, 0, f"Content Briefs — {country}", title_fmt)
        for i, col in enumerate(df_briefs.columns):
            ws3.write(1, i, col, hdr)
        ws3.set_column("D:D", 40)
        ws3.set_column("E:E", 60)

        # Sheet 4: Blog Outline
        outline_rows = []
        h2_counter = 0
        for section in blog_outline:
            if section["level"] in ["H1", "H2"]:
                h2_counter += 1
            outline_rows.append({
                "section_order": h2_counter,
                "level": section["level"],
                "heading": section["heading"],
                "section_type": section["type"],
                "target_keywords": " | ".join(section["keywords"][:6]),
                "total_impressions": section["impressions"],
                "writer_notes": section["notes"],
                "status": "",
                "assigned_to": "",
            })
        df_outline = pd.DataFrame(outline_rows)
        df_outline.to_excel(writer, sheet_name="Blog Outline", index=False, startrow=1)
        ws4 = writer.sheets["Blog Outline"]
        ws4.write(0, 0, f"Blog Outline — {blog_title}", title_fmt)
        for i, col in enumerate(df_outline.columns):
            ws4.write(1, i, col, hdr)

        for row_idx, row_data in df_outline.iterrows():
            excel_row = row_idx + 2
            level = row_data["level"]
            stype = str(row_data.get("section_type", ""))
            fmt = None
            if level == "H1":
                fmt = h1_fmt
            elif level == "H2" and "NEW" in stype:
                fmt = new_section_fmt
            elif level == "H2" and "no GSC signal" in stype:
                fmt = gap_fmt
            if fmt:
                for col_idx in range(len(df_outline.columns)):
                    val = df_outline.iloc[row_idx, col_idx]
                    ws4.write(excel_row, col_idx, val if pd.notna(val) else "", fmt)

        ws4.set_column("A:A", 12)
        ws4.set_column("B:B", 6)
        ws4.set_column("C:C", 50)
        ws4.set_column("D:D", 24)
        ws4.set_column("E:E", 65)
        ws4.set_column("F:F", 15)
        ws4.set_column("G:G", 70)
        ws4.freeze_panes(2, 3)

        # Sheet 5: Gap Analysis
        if gap_clusters:
            gap_rows = []
            for entry in gap_clusters:
                gap_rows.append({
                    "cluster_id": entry["cluster_id"],
                    "cluster_name": entry["cluster_name"],
                    "keyword_count": entry["keyword_count"],
                    "total_impressions": entry["total_impressions"],
                    "opportunity_score": entry["opportunity_score"],
                    "closest_h2": entry["best_h2_match"],
                    "similarity": entry["similarity"],
                    "sample_keywords": " | ".join(entry["top_keywords"][:5]),
                    "recommendation": "Add as new H2" if entry["total_impressions"] >= 100 else "Consider adding or merging",
                })
            df_gaps = pd.DataFrame(gap_rows)
            df_gaps.to_excel(writer, sheet_name="Gap Analysis", index=False, startrow=1)
            ws5 = writer.sheets["Gap Analysis"]
            ws5.write(0, 0, "Content Gaps — Clusters NOT Covered", title_fmt)
            for i, col in enumerate(df_gaps.columns):
                ws5.write(1, i, col, hdr)
            ws5.set_column("B:B", 30)
            ws5.set_column("H:H", 60)
            ws5.set_column("I:I", 25)

    output.seek(0)
    return output.getvalue()


# ============================================================
# MAIN WORKFLOW
# ============================================================

# --- Step 1: Upload ---
st.header("1️⃣ Upload GSC Export")

col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        f"Upload GSC export for **{country}**",
        type=["xlsx", "xls", "csv", "tsv"],
        help="Export from Search Console → Performance → Queries → Download"
    )
with col_info:
    st.info(
        "**Expected columns:**\n\n"
        "• `Top queries`\n"
        "• `Clicks`\n"
        "• `Impressions`\n"
        "• `CTR`\n"
        "• `Position`"
    )

if uploaded_file is None:
    st.warning("👆 Upload a GSC export file to begin.")
    st.stop()

# Parse the file
try:
    df_raw = parse_gsc_file(uploaded_file)
except Exception as e:
    st.error(f"Error parsing file: {e}")
    st.stop()

st.success(f"✅ Loaded **{len(df_raw)}** rows from `{uploaded_file.name}`")

with st.expander("Preview raw data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

# Normalize columns
try:
    df, detected = normalize_gsc(df_raw, min_impressions, country)
except Exception as e:
    st.error(f"Error normalizing data: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Unique queries", f"{len(df):,}")
if "impressions" in df.columns:
    col2.metric("Total impressions", f"{int(df['impressions'].sum()):,}")
if "clicks" in df.columns:
    col3.metric("Total clicks", f"{int(df['clicks'].sum()):,}")
if "position" in df.columns:
    col4.metric("Avg position", f"{df['position'].mean():.1f}")

with st.expander("Detected columns"):
    st.json(detected)


# --- Step 2: Existing blog H2s ---
st.header("2️⃣ Existing Blog Structure")
st.caption(
    f"Paste the current H2 headings of the existing blog (one per line). "
    f"This is used to detect gaps between search demand and current content."
)

default_h2s = """Artificial Intelligence and Machine Learning
Cybersecurity
Cloud Computing
Data Science and Analytics
Project Management
Software Development
Digital Marketing
Healthcare and Nursing
Financial Analysis
Engineering
English Language Teaching
Green Energy and Sustainability"""

h2_text = st.text_area(
    "Existing H2 sections",
    value=default_h2s,
    height=240,
    help="One heading per line. Leave defaults for Edstellar 'In-Demand Skills in Australia' blog."
)
existing_h2s = [line.strip() for line in h2_text.split("\n") if line.strip()]

st.caption(f"**{len(existing_h2s)}** H2 sections loaded")


# --- Step 3: Run analysis ---
st.header("3️⃣ Run Analysis")

run = st.button("🚀 Run Clustering & Generate Outline", type="primary", use_container_width=True)

if not run and "analysis_done" not in st.session_state:
    st.info("Click **Run Clustering & Generate Outline** to process the data.")
    st.stop()

if run:
    # Clear previous results
    for key in ["df_clustered", "embeddings", "cluster_summary", "covered_clusters",
                "gap_clusters", "h2_to_clusters", "cluster_keyword_map", "blog_outline",
                "analysis_done"]:
        st.session_state.pop(key, None)

    # Load model
    with st.spinner(f"Loading embedding model `{embedding_model_name}`..."):
        model = load_embedding_model(embedding_model_name)

    # Generate embeddings
    progress = st.progress(0, text="Generating semantic embeddings...")
    embeddings = model.encode(df["query"].tolist(), show_progress_bar=False, batch_size=128)
    progress.progress(30, text="Running UMAP + HDBSCAN clustering...")

    # Cluster
    df_clustered = cluster_queries(df, embeddings, min_cluster_size, min_samples, umap_n_neighbors)
    progress.progress(60, text="Labeling clusters with TF-IDF...")

    # Label clusters
    cluster_labels_map = {}
    for cid in sorted(df_clustered["cluster_id"].unique()):
        if cid == -1:
            cluster_labels_map[cid] = "Unclustered"
            continue
        kws = df_clustered[df_clustered["cluster_id"] == cid]["query"].tolist()
        cluster_labels_map[cid] = label_cluster(kws, country)
    df_clustered["cluster_name"] = df_clustered["cluster_id"].map(cluster_labels_map)

    progress.progress(80, text="Scoring clusters & finding gaps...")

    # Summary
    cluster_summary = build_cluster_summary(df_clustered, country)

    # Gap analysis
    covered, gaps, h2_map, kw_map = analyze_gaps(
        df_clustered, embeddings, cluster_summary, existing_h2s, model, match_threshold
    )

    progress.progress(95, text="Building blog outline...")

    # Blog outline
    outline = build_blog_outline(
        country=country,
        year=int(target_year),
        existing_h2s=existing_h2s,
        h2_to_clusters=h2_map,
        gap_clusters=gaps,
        cluster_keyword_map=kw_map,
        df=df_clustered,
    )

    progress.progress(100, text="Done!")
    progress.empty()

    # Save to session state
    st.session_state["df_clustered"] = df_clustered
    st.session_state["embeddings"] = embeddings
    st.session_state["cluster_summary"] = cluster_summary
    st.session_state["covered_clusters"] = covered
    st.session_state["gap_clusters"] = gaps
    st.session_state["h2_to_clusters"] = h2_map
    st.session_state["cluster_keyword_map"] = kw_map
    st.session_state["blog_outline"] = outline
    st.session_state["analysis_done"] = True

# Retrieve from session state
df_clustered = st.session_state.get("df_clustered")
cluster_summary = st.session_state.get("cluster_summary")
covered_clusters = st.session_state.get("covered_clusters", [])
gap_clusters = st.session_state.get("gap_clusters", [])
blog_outline = st.session_state.get("blog_outline", [])

if df_clustered is None:
    st.stop()


# --- Step 4: Results ---
st.header("4️⃣ Results")

n_clusters = cluster_summary["cluster_id"].nunique() if len(cluster_summary) > 0 else 0
n_noise = (df_clustered["cluster_id"] == -1).sum()
n_gaps = len(gap_clusters)
n_covered = len(covered_clusters)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clusters found", n_clusters)
m2.metric("Queries clustered", f"{len(df_clustered) - n_noise:,}")
m3.metric("Covered by blog", n_covered, help="Clusters matching an existing H2")
m4.metric("Content gaps", n_gaps, delta=f"{n_gaps} new sections" if n_gaps else None, delta_color="normal")


# --- Tabs for different views ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Cluster Map",
    "📋 Cluster Summary",
    "🔴 Gap Analysis",
    "📝 Blog Outline",
    "💾 Export"
])

# -------- TAB 1: Cluster Map --------
with tab1:
    st.subheader("Semantic map of search queries")
    st.caption("Each dot = one query. Clusters are positioned by semantic similarity. Size = impressions.")

    df_viz = df_clustered[df_clustered["cluster_id"] != -1].copy()
    df_viz["label"] = df_viz["cluster_id"].astype(str) + ": " + df_viz["cluster_name"]
    if "impressions" in df_viz.columns:
        df_viz["log_impressions"] = np.log1p(df_viz["impressions"])
        size_col = "log_impressions"
    else:
        size_col = None

    hover_cols = {"query": True, "cluster_name": True, "x": False, "y": False, "label": False}
    if size_col:
        hover_cols["log_impressions"] = False
    for m in ["clicks", "impressions", "position", "ctr"]:
        if m in df_viz.columns:
            hover_cols[m] = ":.2f" if m in ["ctr", "position"] else True

    fig1 = px.scatter(
        df_viz, x="x", y="y", color="label", size=size_col, size_max=18,
        hover_data=hover_cols,
        title=f"Search Query Clusters — {country}",
        height=650,
        color_discrete_sequence=px.colors.qualitative.Set3 + px.colors.qualitative.Pastel,
    )
    fig1.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#c9d1d9",
        legend=dict(font=dict(size=9), title="Cluster", itemsizing="constant"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    )
    fig1.update_traces(marker=dict(line=dict(width=0.3, color="#30363d")))
    st.plotly_chart(fig1, use_container_width=True)

    # Opportunity ranking
    st.subheader("Top content opportunities")
    if len(cluster_summary) > 0:
        top_n = min(20, len(cluster_summary))
        top = cluster_summary.head(top_n).iloc[::-1]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=top["cluster_name"],
            x=top["opportunity_score"],
            orientation="h",
            marker_color=["#f97316" if s >= top["opportunity_score"].quantile(0.75) else "#3b82f6"
                          for s in top["opportunity_score"]],
            text=[f"{row['keyword_count']} kws · {row.get('total_impressions', 0):,} impr"
                  for _, row in top.iterrows()],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br>%{text}<extra></extra>",
        ))
        fig2.update_layout(
            title=f"Top {top_n} Content Opportunities — {country}",
            xaxis_title="Opportunity Score",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#c9d1d9",
            height=max(400, top_n * 32),
            margin=dict(l=250, r=120),
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------- TAB 2: Cluster Summary --------
with tab2:
    st.subheader("All clusters ranked by opportunity")
    display_cols = [
        "cluster_id", "cluster_name", "keyword_count",
        "total_impressions", "total_clicks",
        "weighted_position", "weighted_ctr",
        "opportunity_score", "search_intent", "top_queries"
    ]
    display_cols = [c for c in display_cols if c in cluster_summary.columns]

    st.dataframe(
        cluster_summary[display_cols],
        use_container_width=True,
        height=500,
        column_config={
            "cluster_name": st.column_config.TextColumn("Cluster Name", width="medium"),
            "total_impressions": st.column_config.NumberColumn("Impressions", format="%d"),
            "total_clicks": st.column_config.NumberColumn("Clicks", format="%d"),
            "weighted_position": st.column_config.NumberColumn("Avg Position", format="%.2f"),
            "weighted_ctr": st.column_config.NumberColumn("Avg CTR", format="%.2f%%"),
            "opportunity_score": st.column_config.NumberColumn("Opp Score", format="%.1f"),
            "top_queries": st.column_config.TextColumn("Sample Queries", width="large"),
        }
    )

    # Drill-down into individual clusters
    st.divider()
    st.subheader("Drill into a cluster")
    cluster_options = {
        f"#{row['cluster_id']} — {row['cluster_name']} ({row['keyword_count']} kws)": row["cluster_id"]
        for _, row in cluster_summary.iterrows()
    }
    selected_label = st.selectbox("Select a cluster", options=list(cluster_options.keys()))
    if selected_label:
        selected_id = cluster_options[selected_label]
        cluster_queries_df = df_clustered[df_clustered["cluster_id"] == selected_id].copy()
        cluster_queries_df = cluster_queries_df.drop(columns=["x", "y", "cluster_id", "cluster_name", "country"],
                                                     errors="ignore")
        if "impressions" in cluster_queries_df.columns:
            cluster_queries_df = cluster_queries_df.sort_values("impressions", ascending=False)
        st.dataframe(cluster_queries_df, use_container_width=True, height=400)

# -------- TAB 3: Gap Analysis --------
with tab3:
    st.subheader("Content gaps — clusters NOT covered by existing blog")
    st.caption(f"Match threshold: {match_threshold}. Lower the threshold in the sidebar to find fewer gaps.")

    if not gap_clusters:
        st.success("🎉 No significant gaps found. All clusters map to existing H2 sections.")
    else:
        for g in gap_clusters:
            with st.expander(
                f"🔴 **{g['cluster_name']}** — {g['keyword_count']} queries, "
                f"{g['total_impressions']:,} impressions"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Keyword count", g["keyword_count"])
                    st.metric("Total impressions", f"{g['total_impressions']:,}")
                with c2:
                    st.metric("Opportunity score", g["opportunity_score"])
                    st.metric("Similarity to closest H2", f"{g['similarity']:.3f}")

                st.write(f"**Closest existing H2:** {g['best_h2_match']} (below match threshold)")
                st.write("**Top keywords in this gap:**")
                for kw in g["top_keywords"]:
                    st.write(f"• {kw}")

    st.divider()
    st.subheader("✅ Existing H2 sections with search validation")
    h2_map = st.session_state.get("h2_to_clusters", {})
    h2_data = []
    for h2, clusters in h2_map.items():
        total_impr = sum(c["total_impressions"] for c in clusters)
        total_kws = sum(c["keyword_count"] for c in clusters)
        h2_data.append({
            "H2 Section": h2,
            "Matched Clusters": len(clusters),
            "Total Keywords": total_kws,
            "Total Impressions": total_impr,
            "Status": "✅ Validated" if clusters else "⚠️ No GSC signal",
        })
    df_h2 = pd.DataFrame(h2_data).sort_values("Total Impressions", ascending=False)
    st.dataframe(df_h2, use_container_width=True, hide_index=True)

# -------- TAB 4: Blog Outline --------
with tab4:
    blog_title = title_template.format(country=country, year=target_year)
    st.subheader(f"📝 {blog_title}")

    n_h2 = sum(1 for s in blog_outline if s["level"] == "H2")
    n_h3 = sum(1 for s in blog_outline if s["level"] == "H3")
    n_new = sum(1 for s in blog_outline if s["level"] == "H2" and "NEW" in str(s.get("type", "")))

    s1, s2, s3 = st.columns(3)
    s1.metric("H2 sections", n_h2)
    s2.metric("H3 sub-sections", n_h3)
    s3.metric("🆕 New sections", n_new, delta="added from gaps" if n_new else None)

    st.divider()

    for section in blog_outline:
        level = section["level"]
        heading = section["heading"]
        stype = section.get("type", "")
        keywords = section.get("keywords", [])
        notes = section.get("notes", "")
        impressions = section.get("impressions", 0)

        if level == "H1":
            st.markdown(f"# {heading}")
        elif level == "H2":
            if "NEW" in stype:
                st.markdown(f"### 🆕 H2: {heading}")
                st.markdown(f"*{stype}*")
            elif "no GSC signal" in stype:
                st.markdown(f"### ⚠️ H2: {heading}")
                st.markdown(f"*{stype}*")
            else:
                st.markdown(f"### H2: {heading}")
                if stype:
                    st.caption(stype)
        elif level == "H3":
            st.markdown(f"**↳ H3: {heading}**")

        if keywords:
            kw_display = " · ".join([f"`{k}`" for k in keywords[:5]])
            st.markdown(f"🎯 **Target keywords:** {kw_display}")

        if impressions > 0:
            st.caption(f"📊 Combined impressions: {impressions:,}")

        if notes:
            st.info(notes)

        st.write("")  # spacer

# -------- TAB 5: Export --------
with tab5:
    st.subheader("Download full report")
    blog_title = title_template.format(country=country, year=target_year)

    st.write(
        "The Excel report includes 5 sheets:\n"
        "1. **Cluster Summary** — all clusters ranked by opportunity\n"
        "2. **Keywords by Cluster** — every query with its assignment\n"
        "3. **Content Briefs** — top 30 clusters with secondary keywords\n"
        "4. **Blog Outline** — full H2/H3 outline with writer notes (color-coded)\n"
        "5. **Gap Analysis** — uncovered clusters with recommendations"
    )

    with st.spinner("Building Excel report..."):
        excel_bytes = build_excel_report(
            country=country,
            blog_title=blog_title,
            df=df_clustered,
            cluster_summary=cluster_summary,
            blog_outline=blog_outline,
            gap_clusters=gap_clusters,
        )

    country_slug = country.lower().replace(" ", "-")
    filename = f"gsc_clusters_{country_slug}.xlsx"

    st.download_button(
        label="📥 Download Excel report",
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    # Markdown export of outline
    st.subheader("Copy outline as Markdown")
    md_lines = [f"# {blog_title}", ""]
    for section in blog_outline:
        level = section["level"]
        heading = section["heading"]
        stype = section.get("type", "")
        keywords = section.get("keywords", [])
        notes = section.get("notes", "")

        if level == "H1":
            continue  # already added
        elif level == "H2":
            prefix = "## 🆕 " if "NEW" in stype else "## "
            md_lines.append(f"{prefix}{heading}")
        elif level == "H3":
            md_lines.append(f"### {heading}")

        if keywords:
            md_lines.append(f"*Target keywords: {', '.join(keywords[:5])}*")
        if notes:
            md_lines.append(f"> {notes}")
        md_lines.append("")

    md_content = "\n".join(md_lines)
    st.code(md_content, language="markdown")
    st.download_button(
        label="📄 Download outline as Markdown",
        data=md_content,
        file_name=f"blog_outline_{country_slug}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Built for country-by-country GSC analysis. "
    "Change the country in the sidebar and upload a new file to analyze a different market."
)
