import streamlit as st
import pandas as pd
import itertools
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from networkx.algorithms import community
from collections import Counter
import nltk
from nltk.corpus import stopwords

# --- Função principal ---
def show():

    # --- Dark mode color settings ---
    DARK_BG = "#222222"
    DARK_FONT = "#ffffff"

    st.title("📊 Science Mapping Dashboard")

    uploaded_file = st.file_uploader("Upload Excel file with columns 'Title',  'Article References' and 'Keywords'", type=["xlsx"])

    if not uploaded_file:
        st.info("Please upload an Excel (.xlsx) file with 'Title', 'Article References' and 'Keywords' columns.")
        return

    df = pd.read_excel(uploaded_file)

    # --- Helper functions ---
    def clean_refs(refs):
        if pd.isna(refs):
            return []
        refs_list = [r.strip() for r in refs.split(';') if r.strip()]
        clean = []
        for r in refs_list:
            if any(c.isalpha() for c in r) and ' ' in r:
                clean.append(r)
            elif r.lower().startswith("10.") or "doi.org" in r.lower():
                clean.append(r)
            else:
                clean.append(r)
        return clean

    def get_orange_color(degree, max_degree):
        norm = degree / max_degree if max_degree > 0 else 0
        r = 255
        g = int(200 - 100 * norm)
        b = int(100 * (1 - norm))
        return f"rgb({r},{g},{b})"
    
    def run_clustering(G, algo="Greedy"):
        if G.number_of_nodes() == 0:
            return []

        if algo == "Greedy":
            return community.greedy_modularity_communities(G)
        elif algo == "Label Propagation":
            return community.asyn_lpa_communities(G)
        elif algo == "Louvain":
            try:
                import community as community_louvain
            except ImportError:
                st.error("Please install python-louvain: pip install python-louvain")
                return []
            partition = community_louvain.best_partition(G)
            clusters = {}
            for node, cluster_id in partition.items():
                clusters.setdefault(cluster_id, []).append(node)
            return [set(c) for c in clusters.values()]
        else:
            return community.greedy_modularity_communities(G)  # fallback
        
    # --- Reference summary metrics ---
    st.subheader("Reference Summary")

    st.markdown("""
    The Co-citation Analysis and Bibliographic Coupling Analysis in this section both depend on reference data present on the file provided.
    Bellow you can find information of the number of references found in your file.
    """)


    total_refs = sum(len(clean_refs(r)) for r in df['Article References'].dropna())
    articles_with_refs = df['Article References'].dropna().shape[0]
    articles_missing_refs = df['Article References'].isna().sum()
    total_articles = df.shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total References Found", total_refs)
    col2.metric("Articles with References", articles_with_refs)
    col3.metric("Articles Missing References", articles_missing_refs)
    col4.metric("Percentage of Articles Missing References", f"{articles_missing_refs/total_articles*100:.2f}%")

    # =====================
    # --- Co-Citation ---
    # =====================
    st.subheader("Co-Citation Analysis")

    all_pairs = []
    for refs in df['Article References'].dropna():
        refs_list = clean_refs(refs)
        refs_list = list(set(refs_list))
        for combo in itertools.combinations(sorted(refs_list), 2):
            all_pairs.append(combo)

    pairs_df = pd.DataFrame(all_pairs, columns=['Ref1', 'Ref2'])
    co_citation_counts = pairs_df.value_counts().reset_index(name='Count')
    top20_df = co_citation_counts.sort_values("Count", ascending=False).head(20)

    st.markdown("**Top 20 Co-Citation Pairs**")
    st.dataframe(top20_df.style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', '#111111'), ('color', DARK_FONT)]},
        {'selector': 'td', 'props': [('background-color', '#222222'), ('color', DARK_FONT)]}
    ]))
    st.download_button("Download Top 20 Co-Citations as CSV", top20_df.to_csv(index=False).encode("utf-8"), "top20_co_citation.csv", "text/csv")

    # --- Build Co-Citation Graph ---
    top_pairs = co_citation_counts.sort_values("Count", ascending=False).head(100)
    G = nx.Graph()
    for _, row in top_pairs.iterrows():
        G.add_edge(row['Ref1'], row['Ref2'], weight=row['Count'])

    algo_choice = st.selectbox("Select Clustering Algorithm (Co-Citation)", ["Greedy", "Louvain", "Label Propagation"])
    clusters = run_clustering(G, algo_choice)

    cluster_dict = {i+1: list(c) for i, c in enumerate(clusters)}
    cluster_options = ["All"] + [f"Cluster {i}" for i in cluster_dict.keys()]
    selected_cluster = st.selectbox("Select Co-Citation Cluster", cluster_options)

    G_vis = Network(height="600px", width="100%", notebook=False, bgcolor=DARK_BG, font_color=DARK_FONT)
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
    st.markdown("**Legend: Node → Reference**")
    st.dataframe(pd.DataFrame(legend_data).style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', '#111111'), ('color', DARK_FONT)]},
        {'selector': 'td', 'props': [('background-color', '#222222'), ('color', DARK_FONT)]}
    ]))

    # =====================
    # --- Bibliographic Coupling ---
    # =====================
    st.subheader("Bibliographic Coupling Analysis")

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
    st.dataframe(top20_bc.style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', '#111111'), ('color', DARK_FONT)]},
        {'selector': 'td', 'props': [('background-color', '#222222'), ('color', DARK_FONT)]}
    ]))
    st.download_button("Download Top 20 Bibliographic Coupling", top20_bc.to_csv(index=False).encode("utf-8"), "top20_bibliographic_coupling.csv", "text/csv")

    # --- Build BC Graph ---
    top_bc = bc_df.head(100)
    G_bc = nx.Graph()
    for _, row in top_bc.iterrows():
        G_bc.add_edge(row['Article1'], row['Article2'], weight=row['Shared_Refs'])

    algo_choice_bc = st.selectbox(
    "Select Clustering Algorithm (Bibliographic Coupling)",
    ["Greedy", "Louvain", "Label Propagation"],
    key="clustering_algo_bc"
    )
    clusters_bc = run_clustering(G_bc, algo_choice_bc)

    cluster_dict_bc = {i+1: list(c) for i, c in enumerate(clusters_bc)}
    cluster_options_bc = ["All"] + [f"Cluster {i}" for i in cluster_dict_bc.keys()]
    selected_cluster_bc = st.selectbox("Select Bibliographic Coupling Cluster", cluster_options_bc)

    G_vis_bc = Network(height="600px", width="100%", notebook=False, bgcolor=DARK_BG, font_color=DARK_FONT)
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
    st.markdown("**Legend: Node → Article (BC)**")
    st.dataframe(pd.DataFrame(legend_data_bc).style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', '#111111'), ('color', DARK_FONT)]},
        {'selector': 'td', 'props': [('background-color', '#222222'), ('color', DARK_FONT)]}
    ]))

    # =====================
    # --- Co-Word Analysis 
    # =====================


    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    st.subheader("Co-Word Analysis (Focus Word)")

        # --- Text fields summary metrics ---
    st.subheader("Text Fields Summary")

    st.markdown("""
    The Co-word Analysis in this section depends on title, keywords and abstract data on the file provided.
    Bellow you can find information on the size of the data for each of these fields found in your file.
    """)

    fields = ['Title', 'Keywords', 'Abstract']
    summary_data = []

    for field in fields:
        total_present = df[field].dropna().shape[0]
        total_missing = df[field].isna().sum()
        total_articles = df.shape[0]
        percentage_missing = total_missing / total_articles * 100
        
        summary_data.append({
            "Field": field,
            "Total Present": total_present,
            "Missing": total_missing,
            "Percentage Missing": f"{percentage_missing:.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)

    # Display nicely in Streamlit
    st.dataframe(summary_df.style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', '#111111'), ('color', 'white')]},
        {'selector': 'td', 'props': [('background-color', '#222222'), ('color', 'white')]}
    ]))

    focus_word = st.text_input("Focus Word", value="future").lower()
    fields = st.multiselect("Select fields to include", ["Title", "Keywords", "Abstract"], default=["Title", "Keywords", "Abstract"])
    top_n = st.slider("Top N co-words to display", min_value=5, max_value=100, value=20, step=5)

    if focus_word and fields:
        # Combine text from selected fields
        texts = df[fields].fillna("").agg(" ".join, axis=1).str.lower()
        
        # Filter only rows containing the focus word
        texts_with_focus = texts[texts.str.contains(focus_word)]
        
        # Tokenize and remove stopwords
        token_lists = [ [w for w in t.split() if w.isalpha() and w not in stop_words] for t in texts_with_focus ]
        
        # Count co-occurrences
        co_word_counter = Counter()
        for tokens in token_lists:
            tokens_set = set(tokens) - {focus_word}
            co_word_counter.update(tokens_set)
        
        # Keep only top N co-words
        top_co_words = dict(co_word_counter.most_common(top_n))
        
        # Build network
        G_word = nx.Graph()
        for w, count in top_co_words.items():
            G_word.add_node(w, size=10 + count)
            G_word.add_edge(focus_word, w, weight=count)
        G_word.add_node(focus_word, size=20)

        # Visualize with PyVis (dark mode)
        G_vis_word = Network(height="600px", width="100%", bgcolor=DARK_BG, font_color=DARK_FONT)
        for node in G_word.nodes():
            G_vis_word.add_node(node, label=node, title=node, size=G_word.nodes[node].get("size", 15))
        for u, v, data in G_word.edges(data=True):
            G_vis_word.add_edge(u, v, value=data['weight'])

        G_vis_word.save_graph("co_word_graph.html")
        with open("co_word_graph.html", 'r', encoding='utf-8') as f:
            HtmlFile = f.read()
        components.html(HtmlFile, height=600)
        st.download_button("Download Co-Word Graph", HtmlFile, "co_word_graph.html", "text/html")
