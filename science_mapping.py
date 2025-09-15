import streamlit as st
import pandas as pd
import itertools
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from networkx.algorithms import community

def show():
    st.title("Bibliometric Analysis - Co-Citation and Bibliographic Coupling")

    uploaded_file = st.file_uploader("Upload Excel file with columns 'Title' and 'Article References'", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        # --- Clean references for new format, prefer Title over DOI ---
        def clean_refs(refs):
            """
            Each reference line can contain: Title; DOI; unstructured text (all separated by ;)
            Preference order:
                1. Title
                2. DOI if no title
                3. raw text if neither
            """
            if pd.isna(refs):
                return []

            refs_list = [r.strip() for r in refs.split(';') if r.strip()]
            clean = []
            for r in refs_list:
                if any(c.isalpha() for c in r) and ' ' in r:
                    clean.append(r)  # Title
                elif r.lower().startswith("10.") or "doi.org" in r.lower():
                    clean.append(r)  # DOI
                else:
                    clean.append(r)  # Fallback text
            return clean

        # --- Reference summary metrics ---
        st.subheader("Reference Summary")
        total_refs = sum(len(clean_refs(r)) for r in df['Article References'].dropna())
        articles_with_refs = df['Article References'].dropna().shape[0]
        articles_missing_refs = df['Article References'].isna().sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total References Found", total_refs)
        col2.metric("Articles with References", articles_with_refs)
        col3.metric("Articles Missing References", articles_missing_refs)

        # =====================
        # --- Co-Citation ---
        # =====================
        all_pairs = []
        for refs in df['Article References'].dropna():
            refs_list = clean_refs(refs)
            refs_list = list(set(refs_list))  # remove duplicates
            for combo in itertools.combinations(sorted(refs_list), 2):
                all_pairs.append(combo)

        pairs_df = pd.DataFrame(all_pairs, columns=['Ref1', 'Ref2'])
        co_citation_counts = pairs_df.value_counts().reset_index(name='Count')

        st.subheader("Top 20 Co-Citation Pairs")
        top20_df = co_citation_counts.sort_values("Count", ascending=False).head(20)
        st.dataframe(top20_df)
        csv_top20 = top20_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 20 Co-Citations as CSV", csv_top20, "top20_co_citation.csv", "text/csv")

        # Build graph
        top_pairs = co_citation_counts.sort_values("Count", ascending=False).head(100)
        G = nx.Graph()
        for _, row in top_pairs.iterrows():
            G.add_edge(row['Ref1'], row['Ref2'], weight=row['Count'])

        clusters = community.greedy_modularity_communities(G)
        cluster_dict = {i+1: list(c) for i, c in enumerate(clusters)}

        cluster_options = ["All"] + [f"Cluster {i}" for i in cluster_dict.keys()]
        selected_cluster = st.selectbox("Select Co-Citation Cluster", cluster_options)

        def get_orange_color(degree, max_degree):
            norm = degree / max_degree if max_degree > 0 else 0
            r = 255
            g = int(200 - 100 * norm)
            b = int(100 * (1 - norm))
            return f"rgb({r},{g},{b})"

        G_vis = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
        nodes_to_show = G.nodes() if selected_cluster == "All" else cluster_dict[int(selected_cluster.split()[1])]
        max_degree = max([G.degree(node) for node in nodes_to_show]) if nodes_to_show else 1

        legend_data = []
        for cluster_id, cluster_nodes in cluster_dict.items():
            if selected_cluster != "All" and cluster_id != int(selected_cluster.split()[1]):
                continue

            sorted_nodes = sorted(cluster_nodes, key=lambda n: G.degree(n), reverse=True)
            for idx, node in enumerate(sorted_nodes, start=1):
                node_number = f"{cluster_id}-{idx}"
                legend_data.append({"Node": node_number, "Reference": node, "Cluster": cluster_id})

                degree = G.degree(node)
                G_vis.add_node(node, label=node_number, title=node,
                               size=15 + degree*5, color=get_orange_color(degree, max_degree), group=cluster_id)

        for u, v, data in G.edges(data=True):
            if u in nodes_to_show and v in nodes_to_show:
                G_vis.add_edge(u, v, value=data['weight'])

        G_vis.save_graph("co_citation_cluster.html")
        with open("co_citation_cluster.html", 'r', encoding='utf-8') as f:
            HtmlFile = f.read()
        components.html(HtmlFile, height=600)
        st.download_button("Download Co-Citation Graph", HtmlFile, "co_citation_graph.html", "text/html")

        st.subheader("Legend: Node → Reference")
        st.dataframe(pd.DataFrame(legend_data))

        st.subheader("Co-Citation Cluster Summary")
        cluster_summary = pd.DataFrame({"Cluster": cluster_dict.keys(),
                                        "Num_Nodes": [len(nodes) for nodes in cluster_dict.values()]})
        st.dataframe(cluster_summary)

        # =====================
        # --- Bibliographic Coupling ---
        # =====================
        st.subheader("Bibliographic Coupling with Clusters")

        pairs_bc = []
        refs_list = df['Article References'].dropna().tolist()
        titles_list = df['Title'].dropna().tolist()

        for idx1, refs1 in enumerate(refs_list):
            refs1_set = set(clean_refs(refs1))
            for idx2 in range(idx1 + 1, len(refs_list)):
                refs2_set = set(clean_refs(refs_list[idx2]))
                shared_refs = refs1_set & refs2_set
                if shared_refs:
                    pairs_bc.append({
                        'Article1': titles_list[idx1],
                        'Article2': titles_list[idx2],
                        'Shared_Refs': len(shared_refs)
                    })

        bc_df = pd.DataFrame(pairs_bc).sort_values('Shared_Refs', ascending=False)
        top20_bc = bc_df.head(20)
        st.dataframe(top20_bc)
        csv_bc = top20_bc.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 20 Bibliographic Coupling", csv_bc, "top20_bibliographic_coupling.csv", "text/csv")

        # Build BC graph
        top_bc = bc_df.head(100)
        G_bc = nx.Graph()
        for _, row in top_bc.iterrows():
            G_bc.add_edge(row['Article1'], row['Article2'], weight=row['Shared_Refs'])

        clusters_bc = community.greedy_modularity_communities(G_bc)
        cluster_dict_bc = {i+1: list(c) for i, c in enumerate(clusters_bc)}

        cluster_options_bc = ["All"] + [f"Cluster {i}" for i in cluster_dict_bc.keys()]
        selected_cluster_bc = st.selectbox("Select Bibliographic Coupling Cluster", cluster_options_bc)

        G_vis_bc = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
        nodes_to_show_bc = G_bc.nodes() if selected_cluster_bc == "All" else cluster_dict_bc[int(selected_cluster_bc.split()[1])]
        max_degree_bc = max([G_bc.degree(node) for node in nodes_to_show_bc]) if nodes_to_show_bc else 1

        legend_data_bc = []
        for cluster_id, cluster_nodes in cluster_dict_bc.items():
            if selected_cluster_bc != "All" and cluster_id != int(selected_cluster_bc.split()[1]):
                continue

            sorted_nodes = sorted(cluster_nodes, key=lambda n: G_bc.degree(n), reverse=True)
            for idx, node in enumerate(sorted_nodes, start=1):
                node_number = f"{cluster_id}-{idx}"
                legend_data_bc.append({"Node": node_number, "Article": node, "Cluster": cluster_id})

                degree = G_bc.degree(node)
                G_vis_bc.add_node(node, label=node_number, title=node,
                                  size=15 + degree*5, color=get_orange_color(degree, max_degree_bc), group=cluster_id)

        for u, v, data in G_bc.edges(data=True):
            if u in nodes_to_show_bc and v in nodes_to_show_bc:
                G_vis_bc.add_edge(u, v, value=data['weight'])

        G_vis_bc.save_graph("bibliographic_coupling_cluster.html")
        with open("bibliographic_coupling_cluster.html", 'r', encoding='utf-8') as f:
            HtmlFile_bc = f.read()
        components.html(HtmlFile_bc, height=600)
        st.download_button("Download Bibliographic Coupling Graph", HtmlFile_bc, "bibliographic_coupling_clusters.html", "text/html")

        st.subheader("Legend: Node → Article (BC)")
        st.dataframe(pd.DataFrame(legend_data_bc))

        st.subheader("Bibliographic Coupling Cluster Summary")
        cluster_summary_bc = pd.DataFrame({"Cluster": cluster_dict_bc.keys(),
                                           "Num_Nodes": [len(nodes) for nodes in cluster_dict_bc.values()]})
        st.dataframe(cluster_summary_bc)

    else:
        st.info("Please upload an Excel (.xlsx) file with 'Title' and 'Article References' columns.")
