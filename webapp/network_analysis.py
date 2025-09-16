# centrality_analysis_fast.py
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import itertools

def show():
    st.title("Interactive Centrality - Fast Version")

    # --- Upload Excel File ---
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("First rows of the file:")
        st.dataframe(df.head())

        # --- Prepare reference pairs for graph ---
        all_pairs = []
        for refs in df['Article References'].dropna():
            refs_list = [r.strip() for r in refs.split(';') if r.strip()]
            cleaned_refs = []
            for r in refs_list:
                parts = r.split(',')
                if len(parts) >= 2:
                    ref_id = parts[0].strip() + " (" + parts[1].strip() + ")"
                    cleaned_refs.append(ref_id)
                else:
                    cleaned_refs.append(r)
            for combo in itertools.combinations(sorted(set(cleaned_refs)), 2):
                all_pairs.append(combo)

        # --- Count pair frequency ---
        pairs_df = pd.DataFrame(all_pairs, columns=['Ref1', 'Ref2'])
        co_citation_counts = pairs_df.value_counts().reset_index(name='Count')

        # --- Filter top pairs for quick graph ---
        top_pairs = co_citation_counts.sort_values("Count", ascending=False).head(200)

        # --- Create graph ---
        G = nx.Graph()
        for _, row in top_pairs.iterrows():
            G.add_edge(row['Ref1'], row['Ref2'], weight=row['Count'])

        st.write(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # --- Calculate centrality metrics only on filtered nodes ---
        with st.spinner("Calculating centrality metrics..."):
            betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            closeness = nx.closeness_centrality(G)

        centrality_df = pd.DataFrame({
            'Node': list(G.nodes()),
            'Betweenness': [betweenness[n] for n in G.nodes()],
            'Eigenvector': [eigenvector[n] for n in G.nodes()],
            'Closeness': [closeness[n] for n in G.nodes()]
        })

        st.subheader("Centrality Table")
        st.dataframe(centrality_df.sort_values(by='Betweenness', ascending=False))

        st.download_button("Download Centrality Table",
                        centrality_df.to_csv(index=False).encode('utf-8'),
                        "centrality.csv", "text/csv")

        # --- Graph visualization with Pyvis ---
        metric_for_size = st.selectbox("Choose node size metric:",
                                    ["Betweenness", "Eigenvector", "Closeness"])

        G_vis = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")

        # --- Highlight top 10 for each metric ---
        top10_betweenness = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
        top10_eigenvector = sorted(eigenvector, key=eigenvector.get, reverse=True)[:10]
        top10_closeness = sorted(closeness, key=closeness.get, reverse=True)[:10]

        for node in G.nodes():
            size = 15 + centrality_df.loc[centrality_df['Node'] == node, metric_for_size].values[0]*50

            # Color by highlight
            if node in top10_betweenness:
                color = "red"
            elif node in top10_eigenvector:
                color = "blue"
            elif node in top10_closeness:
                color = "green"
            else:
                color = "lightgray"

            border = 5 if node in top10_betweenness + top10_eigenvector + top10_closeness else 1

            G_vis.add_node(node, label=node, size=size, color=color, borderWidth=border,
                        title=f"Betweenness: {betweenness[node]:.4f}\n"
                              f"Eigenvector: {eigenvector[node]:.4f}\n"
                              f"Closeness: {closeness[node]:.4f}")

        for u, v, data in G.edges(data=True):
            G_vis.add_edge(u, v, value=data['weight'])

        # --- Render graph in Streamlit ---
        G_vis.save_graph("centrality_graph_fast.html")
        HtmlFile = open("centrality_graph_fast.html", 'r', encoding='utf-8').read()
        components.html(HtmlFile, height=600)
