import itertools
from collections import Counter
from pathlib import Path

import networkx as nx
import nltk
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from networkx.algorithms import community
from nltk.corpus import stopwords
from pyvis.network import Network


DARK_BG = "#222222"
DARK_FONT = "#ffffff"


def show():
    st.title("📊 Science Mapping Dashboard")

    df = upload_file("science")
    if df is None:
        # Instead of st.stop() just show a message and return
        st.info("Please upload a file to use this section.")
        return 

    display_reference_summary(df)
    display_cocitation_analysis(df)
    display_bibliographic_coupling_analysis(df)
    display_coword_analysis(df)

def safe_download(component_fn, *args, **kwargs):
    try:
        component_fn(*args, **kwargs)
    except (ValueError, st.errors.StreamlitAPIException):
        st.warning("Nothing to download yet (or duplicate key).")


def get_orange_color(degree: int, max_degree: int) -> str:
    norm = degree / max_degree if max_degree else 0
    return f"rgb(255,{int(200 - 100 * norm)},{int(100 * (1 - norm))})"


def clean_refs(refs) -> list:
    try:
        if pd.isna(refs):
            return []
        refs_list = [r.strip() for r in refs.split(";") if r.strip()]
        valid = []
        for r in refs_list:
            if (any(c.isalpha() for c in r) and " " in r) or r.lower().startswith(
                "10."
            ) or "doi.org" in r.lower():
                valid.append(r)
        return valid
    except Exception as exc:
        st.warning(f"Could not parse references: {exc}")
        return []


def upload_file(key):
    uploaded_file = st.file_uploader(
        "Upload Excel file with columns 'Title', 'Article References', "
        "'Keywords' and 'Abstract'",
        type=["xlsx"],
        key=key,
    )

    if uploaded_file is None:
        st.info("Please upload an Excel (.xlsx) file.")
        return None

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the Excel file: {exc}")
        return None

    expected = {"Title", "Article References", "Keywords", "Abstract"}
    missing = expected - set(df.columns)
    if missing:
        st.error("Missing column(s): " + ", ".join(missing))
        return None

    return df



def display_reference_summary(df):
    if "Article References" not in df.columns:
        st.warning("Column 'Article References' is missing – skipping summary.")
        return

    st.subheader("Reference Summary")

    total_refs = sum(len(clean_refs(r)) for r in df["Article References"].dropna())
    articles_with_refs = df["Article References"].notna().sum()
    articles_missing_refs = df["Article References"].isna().sum()
    total_articles = len(df)

    cols = st.columns(4)
    cols[0].metric("Total References Found", total_refs)
    cols[1].metric("Articles with References", articles_with_refs)
    cols[2].metric("Articles Missing References", articles_missing_refs)
    cols[3].metric(
        "Percentage of Articles Missing References",
        f"{articles_missing_refs/total_articles*100:.2f}%",
    )



def display_cocitation_analysis(df):
    if df["Article References"].dropna().empty:
        st.info("No reference data – co-citation analysis skipped.")
        return

    pairs_df = co_citation_pairs_df(df)
    co_citation_counts = pairs_df.value_counts().reset_index(name="Count")

    if co_citation_counts.empty:
        st.info("Not enough data to build co-citation pairs.")
        return

    st.subheader("Co-Citation Analysis")
    display_top_20_cocitation_pairs_table(co_citation_counts)

    G = co_citation_graph(co_citation_counts)
    if G.number_of_nodes() == 0:
        st.warning("Co-citation graph is empty.")
        return

    cluster_dict = cluster_based_on_algo_selected(G, "cocitation")
    html_graph = display_selected_cluster(
        select_cluster_option(cluster_dict, "cocitation"), cluster_dict, G
    )
    if html_graph:
        safe_download(
            st.download_button,
            "Download Co-Citation Graph",
            html_graph,
            "co_citation_graph.html",
            "text/html",
            key="cocitation_download",
        )


def co_citation_pairs_df(df):
    all_pairs = []
    for refs in df["Article References"].dropna():
        rlist = list(set(clean_refs(refs)))
        for combo in itertools.combinations(sorted(rlist), 2):
            all_pairs.append(combo)
    return pd.DataFrame(all_pairs, columns=["Ref1", "Ref2"])


def co_citation_graph(co_citation_counts):
    # Take only the top 100 pairs for performance
    top_pairs = co_citation_counts.sort_values("Count", ascending=False).head(100)
    G = nx.Graph()
    for _, row in top_pairs.iterrows():
        G.add_edge(row["Ref1"], row["Ref2"], weight=row["Count"])
    return G


def display_top_20_cocitation_pairs_table(co_citation_counts):
    top20 = co_citation_counts.sort_values("Count", ascending=False).head(20)
    st.markdown("**Top 20 Co-Citation Pairs**")
    st.dataframe(
        top20.style.set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#111111"), ("color", DARK_FONT)]},
                {"selector": "td", "props": [("background-color", "#222222"), ("color", DARK_FONT)]},
            ]
        )
    )
    safe_download(
        st.download_button,
        "Download Top 20 Co-Citations as CSV",
        top20.to_csv(index=False).encode("utf-8"),
        "top20_co_citation.csv",
        "text/csv",
        key="top20_co_citation",
    )


def display_bibliographic_coupling_analysis(df):
    if df["Article References"].dropna().empty:
        st.info("No reference data – bibliographic coupling skipped.")
        return

    bc_pairs_df = bibliographic_coupling_pairs(df)
    if bc_pairs_df.empty:
        st.info("Not enough overlap to build bibliographic-coupling pairs.")
        return

    st.subheader("Bibliographic Coupling Analysis")
    display_top_20_bc_pairs_table(bc_pairs_df)

    G = bc_graph(bc_pairs_df)
    if G.number_of_nodes() == 0:
        st.warning("Bibliographic-coupling graph is empty.")
        return

    cluster_dict = cluster_based_on_algo_selected(G, "bc")
    html_graph = display_selected_cluster(select_cluster_option(cluster_dict, "bc"), cluster_dict, G)
    if html_graph:
        safe_download(
            st.download_button,
            "Download Bibliographic Coupling Graph",
            html_graph,
            "bibliographic_coupling_graph.html",
            "text/html",
            key="bc_download",
        )


def bibliographic_coupling_pairs(df):
    pairs_bc = []
    refs_list = df["Article References"].dropna().tolist()
    titles_list = df["Title"].fillna("Untitled").tolist()

    for idx1, refs1 in enumerate(refs_list):
        refs1_set = set(clean_refs(refs1))
        for idx2 in range(idx1 + 1, len(refs_list)):
            shared_refs = refs1_set & set(clean_refs(refs_list[idx2]))
            if shared_refs:
                pairs_bc.append(
                    {
                        "Article1": titles_list[idx1],
                        "Article2": titles_list[idx2],
                        "Shared_Refs": len(shared_refs),
                    }
                )
    return pd.DataFrame(pairs_bc).sort_values("Shared_Refs", ascending=False)


def bc_graph(bc_df):
    top_bc = bc_df.head(100)
    G = nx.Graph()
    for _, row in top_bc.iterrows():
        G.add_edge(row["Article1"], row["Article2"], weight=row["Shared_Refs"])
    return G


def display_top_20_bc_pairs_table(bc_df):
    top20_bc = bc_df.head(20)
    st.markdown("**Top 20 Bibliographic-Coupling Pairs**")
    st.dataframe(
        top20_bc.style.set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#111111"), ("color", DARK_FONT)]},
                {"selector": "td", "props": [("background-color", "#222222"), ("color", DARK_FONT)]},
            ]
        )
    )
    safe_download(
        st.download_button,
        "Download Top 20 Bibliographic Coupling",
        top20_bc.to_csv(index=False).encode("utf-8"),
        "top20_bibliographic_coupling.csv",
        "text/csv",
        key="top20_bc",
    )


def display_coword_analysis(df):
    st.subheader("Co-Word Analysis (Focus Word)")
    display_metrics_summary(df)

    focus_word = st.text_input("Focus Word", value="future").lower()
    fields = st.multiselect(
        "Select fields to include",
        ["Title", "Keywords", "Abstract"],
        default=["Title", "Keywords", "Abstract"],
    )
    top_n = st.slider("Top N co-words to display", 5, 100, 20, step=5)

    if not focus_word:
        st.info("Insert a focus word to start the analysis.")
        return
    if not fields:
        st.warning("Select at least one field.")
        return

    display_coword_graph(focus_word, fields, df, top_n)


def display_metrics_summary(df):
    fields = ["Title", "Keywords", "Abstract"]
    summary = []
    total_articles = len(df)

    for fld in fields:
        present = df[fld].notna().sum() if fld in df.columns else 0
        missing = total_articles - present
        summary.append(
            {
                "Field": fld,
                "Total Present": present,
                "Missing": missing,
                "Percentage Missing": f"{missing/total_articles*100:.2f}%",
            }
        )

    st.dataframe(
        pd.DataFrame(summary).style.set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#111111"), ("color", DARK_FONT)]},
                {"selector": "td", "props": [("background-color", "#222222"), ("color", DARK_FONT)]},
            ]
        )
    )


def display_coword_graph(focus_word, fields, df, top_n):
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        with st.spinner("Downloading NLTK stopwords…"):
            nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))

    text_series = df[fields].fillna("").agg(" ".join, axis=1).str.lower()
    subset = text_series[text_series.str.contains(focus_word, regex=False)]

    if subset.empty:
        st.info(f"No occurrences of '{focus_word}' found in the selected fields.")
        return

    token_lists = [
        [w for w in txt.split() if w.isalpha() and w not in stop_words]
        for txt in subset
    ]

    co_counter = Counter()
    for tokens in token_lists:
        co_counter.update(set(tokens) - {focus_word})

    if not co_counter:
        st.info("No co-words found with current parameters.")
        return

    top_words = dict(co_counter.most_common(top_n))

    G = nx.Graph()
    G.add_node(focus_word, size=30)
    for w, cnt in top_words.items():
        G.add_node(w, size=10 + cnt)
        G.add_edge(focus_word, w, weight=cnt)

    G_vis = Network(height="600px", width="100%", bgcolor=DARK_BG, font_color=DARK_FONT)
    for node in G.nodes():
        G_vis.add_node(
            node,
            label=node,
            title=node if node == focus_word else f"{node} ({co_counter[node]}×)",
            size=G.nodes[node]["size"],
        )
    for u, v, data in G.edges(data=True):
        G_vis.add_edge(u, v, value=data["weight"])

    html_path = Path("co_word_graph.html")
    G_vis.save_graph(str(html_path))

    with html_path.open("r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=600)
    safe_download(
        st.download_button,
        "Download Co-Word Graph",
        html,
        "co_word_graph.html",
        "text/html",
        key="coword_download",
    )

def cluster_based_on_algo_selected(G, key_prefix=""):
    algo = st.selectbox(
        "Select Clustering Algorithm",
        ["Greedy", "Louvain", "Label Propagation"],
        key=f"{key_prefix}_algo",
    )
    clusters = run_clustering(G, algo)
    return {i + 1: list(c) for i, c in enumerate(clusters)}


def select_cluster_option(cluster_dict, key_prefix=""):
    options = ["All"] + [f"Cluster {i}" for i in cluster_dict.keys()]
    return st.selectbox("Select Cluster", options, key=f"{key_prefix}_cluster")


def run_clustering(G, algo="Greedy"):
    if G.number_of_nodes() == 0:
        return []

    if algo == "Greedy":
        return community.greedy_modularity_communities(G)

    if algo == "Label Propagation":
        return community.asyn_lpa_communities(G)

    if algo == "Louvain":
        try:
            import community as community_louvain
        except ImportError:
            st.error("Please install python-louvain (`pip install python-louvain`).")
            return []
        partition = community_louvain.best_partition(G)
        clusters = {}
        for node, cid in partition.items():
            clusters.setdefault(cid, []).append(node)
        return [set(c) for c in clusters.values()]

    # fallback
    return community.greedy_modularity_communities(G)


def display_selected_cluster(selected_cluster, cluster_dict, G):
    G_vis = Network(
        height="600px", width="100%", notebook=False, bgcolor=DARK_BG, font_color=DARK_FONT
    )

    nodes_to_show = (
        G.nodes()
        if selected_cluster == "All"
        else cluster_dict[int(selected_cluster.split()[1])]
    )

    max_degree = max((G.degree(n) for n in nodes_to_show), default=1)
    legend_data = []

    for cluster_id, cluster_nodes in cluster_dict.items():
        if selected_cluster != "All" and cluster_id != int(selected_cluster.split()[1]):
            continue

        for idx, node in enumerate(
            sorted(cluster_nodes, key=lambda n: G.degree(n), reverse=True), 1
        ):
            node_number = f"{cluster_id}-{idx}"
            legend_data.append(
                {"Node": node_number, "Reference": node, "Cluster": cluster_id}
            )
            deg = G.degree(node)
            G_vis.add_node(
                node,
                label=node_number,
                title=node,
                size=15 + deg * 5,
                color=get_orange_color(deg, max_degree),
                group=cluster_id,
            )

    for u, v, data in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            G_vis.add_edge(u, v, value=data["weight"])

    html_path = Path("cluster_graph.html")
    G_vis.save_graph(str(html_path))

    with html_path.open("r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=600)
    display_cluster_table(legend_data)
    return html


def display_cluster_table(legend_data):
    st.markdown("**Legend: Node → Reference**")
    st.dataframe(
        pd.DataFrame(legend_data).style.set_table_styles(
            [
                {"selector": "thead", "props": [("background-color", "#111111"), ("color", DARK_FONT)]},
                {"selector": "td", "props": [("background-color", "#222222"), ("color", DARK_FONT)]},
            ]
        )
    )