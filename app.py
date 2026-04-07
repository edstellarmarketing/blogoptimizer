"""
GSC Search Terms → Content Cluster Engine (Streamlit)
======================================================
Upload a Google Search Console export and cluster queries semantically
into actionable content topics.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import os
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
    "find your best content opportunities based on what people are actually searching for."
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    country = st.text_input("Country / Market", value="Australia")

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


# ============================================================
# CACHED MODEL LOADER
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_col(df: pd.DataFrame, candidates: list):
    cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(" ", "_")
        for col_key, col_original in cols_lower.items():
            if key in col_key or col_key in key:
                return col_original
    return None


def parse_gsc_file(uploaded_file) -> pd.DataFrame:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    elif ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_gsc(df_raw: pd.DataFrame, min_impressions: int, country: str):
    query_col = find_col(df_raw, [
        "top queries", "query", "queries", "keyword", "search_term",
        "search term", "top_queries", "keyphrase"
    ])
    clicks_col = find_col(df_raw, ["clicks", "click"])
    impressions_col = find_col(df_raw, ["impressions", "impression", "impr"])
    ctr_col = find_col(df_raw, ["ctr", "click_through_rate", "click through rate"])
    position_col = find_col(df_raw, ["position", "avg_position", "average position", "rank"])

    if query_col is None:
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
                    umap_n_neighbors: int) -> pd.DataFrame:
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


def build_excel_report(country: str, df: pd.DataFrame,
                       cluster_summary: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr = wb.add_format({
            "bold": True, "bg_color": "#1e3a5f", "font_color": "#ffffff",
            "border": 1, "text_wrap": True, "valign": "vcenter", "font_size": 10
        })
        title_fmt = wb.add_format({"bold": True, "font_size": 14, "font_color": "#1e3a5f"})

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

    output.seek(0)
    return output.getvalue()


# ============================================================
# MAIN WORKFLOW
# ============================================================

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

try:
    df_raw = parse_gsc_file(uploaded_file)
except Exception as e:
    st.error(f"Error parsing file: {e}")
    st.stop()

st.success(f"✅ Loaded **{len(df_raw)}** rows from `{uploaded_file.name}`")

with st.expander("Preview raw data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

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


# --- Step 2: Run analysis ---
st.header("2️⃣ Run Clustering")

run = st.button("🚀 Cluster Search Queries", type="primary", use_container_width=True)

if not run and "analysis_done" not in st.session_state:
    st.info("Click **Cluster Search Queries** to process the data.")
    st.stop()

if run:
    for key in ["df_clustered", "cluster_summary", "analysis_done"]:
        st.session_state.pop(key, None)

    with st.spinner(f"Loading embedding model `{embedding_model_name}`..."):
        model = load_embedding_model(embedding_model_name)

    progress = st.progress(0, text="Generating semantic embeddings...")
    embeddings = model.encode(df["query"].tolist(), show_progress_bar=False, batch_size=128)
    progress.progress(40, text="Running UMAP + HDBSCAN clustering...")

    df_clustered = cluster_queries(df, embeddings, min_cluster_size, min_samples, umap_n_neighbors)
    progress.progress(70, text="Labeling clusters with TF-IDF...")

    cluster_labels_map = {}
    for cid in sorted(df_clustered["cluster_id"].unique()):
        if cid == -1:
            cluster_labels_map[cid] = "Unclustered"
            continue
        kws = df_clustered[df_clustered["cluster_id"] == cid]["query"].tolist()
        cluster_labels_map[cid] = label_cluster(kws, country)
    df_clustered["cluster_name"] = df_clustered["cluster_id"].map(cluster_labels_map)

    progress.progress(90, text="Scoring clusters by opportunity...")
    cluster_summary = build_cluster_summary(df_clustered, country)

    progress.progress(100, text="Done!")
    progress.empty()

    st.session_state["df_clustered"] = df_clustered
    st.session_state["cluster_summary"] = cluster_summary
    st.session_state["analysis_done"] = True

df_clustered = st.session_state.get("df_clustered")
cluster_summary = st.session_state.get("cluster_summary")

if df_clustered is None:
    st.stop()


# --- Step 3: Results ---
st.header("3️⃣ Results")

n_clusters = cluster_summary["cluster_id"].nunique() if len(cluster_summary) > 0 else 0
n_noise = (df_clustered["cluster_id"] == -1).sum()
n_clustered = len(df_clustered) - n_noise

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clusters found", n_clusters)
m2.metric("Queries clustered", f"{n_clustered:,}")
m3.metric("Unclustered (noise)", f"{n_noise:,}")
if len(cluster_summary) > 0:
    m4.metric("Top opportunity score", f"{cluster_summary['opportunity_score'].iloc[0]:.1f}")


tab1, tab2, tab3 = st.tabs([
    "📊 Cluster Map",
    "📋 Cluster Summary",
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
        cluster_queries_df = cluster_queries_df.drop(
            columns=["x", "y", "cluster_id", "cluster_name", "country"],
            errors="ignore"
        )
        if "impressions" in cluster_queries_df.columns:
            cluster_queries_df = cluster_queries_df.sort_values("impressions", ascending=False)
        st.dataframe(cluster_queries_df, use_container_width=True, height=400)

# -------- TAB 3: Export --------
with tab3:
    st.subheader("Download full report")
    st.write(
        "The Excel report includes 3 sheets:\n"
        "1. **Cluster Summary** — all clusters ranked by opportunity\n"
        "2. **Keywords by Cluster** — every query with its cluster assignment\n"
        "3. **Content Briefs** — top 30 clusters with primary + secondary keywords"
    )

    with st.spinner("Building Excel report..."):
        excel_bytes = build_excel_report(
            country=country,
            df=df_clustered,
            cluster_summary=cluster_summary,
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


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Built for country-by-country GSC analysis. "
    "Change the country in the sidebar and upload a new file to analyze a different market."
)
