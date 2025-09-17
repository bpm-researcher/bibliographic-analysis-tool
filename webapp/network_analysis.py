import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from networkx.algorithms import community
import science_mapping as sm
from networkx.exception import PowerIterationFailedConvergence
import tempfile, os



def show():
    st.title("Network Analysis (Centrality & Clusters)")

    df = sm.upload_file("network")
    if df is None:
        st.info("Please upload a file to use this section.")
        return

    sm.display_reference_summary(df)
    display(df)


def calculate_metrics_df(G):
    with st.spinner("Calculating centrality metrics…"):
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight",
                                                    normalized=True)
            eigenvector = nx.eigenvector_centrality(G, weight="weight",
                                                    max_iter=1000)
            closeness = nx.closeness_centrality(G)
        except PowerIterationFailedConvergence:
            st.warning("Eigenvector centrality did not converge. "
                       "Setting all values to 0.")
            eigenvector = {n: 0 for n in G.nodes()}

    metrics_df = pd.DataFrame({
        "Node":        list(G.nodes()),
        "Betweenness": [betweenness[n] for n in G.nodes()],
        "Eigenvector": [eigenvector[n]  for n in G.nodes()],
        "Closeness":   [closeness[n]    for n in G.nodes()]
    })
    return metrics_df, betweenness, eigenvector, closeness


# ────────────────────────────────────────────────────────────────────────────────
# CLUSTERS
# ────────────────────────────────────────────────────────────────────────────────
def get_clusters(G):
    """Greedy modularity, deterministic order."""
    if G.number_of_nodes() == 0:
        return {}

    clusters_raw = community.greedy_modularity_communities(G)
    clusters_sorted = sorted(
        [sorted(list(c)) for c in clusters_raw],
        key=lambda lst: (-len(lst), lst[0])
    )
    return {i + 1: c for i, c in enumerate(clusters_sorted)}


def select_cluster_option(cluster_dict, key_prefix=""):
    options = ["All"]
    if cluster_dict:
        options.extend([f"Cluster {i}" for i in cluster_dict])
    return st.selectbox("Select cluster to display",
                        options,
                        key=f"{key_prefix}_cluster")


def display(df):
    pairs_df = sm.co_citation_pairs_df(df)
    co_citation_counts = pairs_df.value_counts().reset_index(name="Count")
    G = sm.co_citation_graph(co_citation_counts)

    if G.number_of_nodes() == 0:
        st.error("Graph has no nodes.")
        st.stop()

    st.write(f"Graph created with {G.number_of_nodes()} nodes and "
             f"{G.number_of_edges()} edges.")

    metrics_df, betw, eig, clo = calculate_metrics_df(G)

    st.subheader("Centrality Table")
    st.dataframe(metrics_df.sort_values("Betweenness", ascending=False))

    st.download_button("Download Centrality Table",
                       metrics_df.to_csv(index=False).encode("utf-8"),
                       file_name="centrality.csv",
                       mime="text/csv")

    display_network_graph(metrics_df, G, betw, eig, clo)


def display_network_graph(metrics_df, G,
                          betweenness, eigenvector, closeness):
    dark_mode = st.toggle("Dark mode", value=False)

    if dark_mode:
        graph_bg       = "#1e1e1e"
        graph_font     = "white"
        default_colour = "#a0a0a0"
        edge_colour    = "#666666"
    else:
        graph_bg       = "#ffffff"
        graph_font     = "black"
        default_colour = "lightgray"
        edge_colour    = "#cccccc"

    # ── Cluster selector ────────────────────────────────────────────────────
    cluster_dict     = get_clusters(G)
    selected_cluster = select_cluster_option(cluster_dict, "cent")

    if selected_cluster == "All":
        nodes_to_show = list(G.nodes())
    else:
        cluster_id    = int(selected_cluster.split()[1])   # "Cluster 3" → 3
        nodes_to_show = cluster_dict.get(cluster_id, [])

    metric_for_size = st.selectbox(
        "Choose node size metric",
        ["Betweenness", "Eigenvector", "Closeness"]
    )

    G_vis = Network(height="600px", width="100%",
                    bgcolor=graph_bg, font_color=graph_font, notebook=False)

    top10_bet = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
    top10_eig = sorted(eigenvector,  key=eigenvector.get,  reverse=True)[:10]
    top10_clo = sorted(closeness,    key=closeness.get,    reverse=True)[:10]

    for node in nodes_to_show:
        size = 15 + metrics_df.loc[
            metrics_df["Node"] == node, metric_for_size
        ].values[0] * 50

        # Highlight top 10s
        if node in top10_bet:
            colour = "#ff4d4d"              # red
        elif node in top10_eig:
            colour = "#4d6aff"              # blue
        elif node in top10_clo:
            colour = "#37b24d"              # green
        else:
            colour = default_colour

        border = 5 if node in (top10_bet + top10_eig + top10_clo) else 1

        G_vis.add_node(
            node,
            label=node,
            size=size,
            color=colour,
            borderWidth=border,
            title=(f"Betweenness: {betweenness[node]:.4f}\n"
                   f"Eigenvector: {eigenvector[node]:.4f}\n"
                   f"Closeness:   {closeness[node]:.4f}")
        )

    for u, v, data in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            G_vis.add_edge(u, v,
                           value=data["weight"],
                           color=edge_colour)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        html_path = tmp.name
    G_vis.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=600)
    os.unlink(html_path)