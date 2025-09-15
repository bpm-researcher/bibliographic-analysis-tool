import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.title("Performance Analysis")

    # Excel file upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file:
        # Load Excel file
        df = pd.read_excel(uploaded_file)

        # --- Total number of unique authors ---
        all_authors = df['Authors'].astype(str).str.split(',').explode().str.strip().unique()
        num_authors = len(all_authors)

        df_authors = df.assign(Authors=df['Authors'].astype(str).str.split(',')) \
                       .explode('Authors')
        df_authors['Authors'] = df_authors['Authors'].str.strip()

        # --- Total and average citations ---
        total_citations = df['Times Cited'].sum()
        avg_citations = df['Times Cited'].mean()

        # --- Display metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Authors", num_authors)
        col2.metric("Total Citations", int(total_citations))
        col3.metric("Average Citations", round(avg_citations, 2))

        # --- Functions to calculate h-index and g-index ---
        def h_index(citations):
            citations = sorted(citations, reverse=True)
            h = sum(c >= i + 1 for i, c in enumerate(citations))
            return h

        def g_index(citations):
            citations = sorted(citations, reverse=True)
            total = 0
            g = 0
            for i, c in enumerate(citations, start=1):
                total += c
                if total >= i**2:
                    g = i
            return g

        # --- Calculate metrics per author ---
        results = []
        for author, group in df_authors.groupby("Authors"):
            citations = group['Times Cited'].fillna(0).astype(int).tolist()
            results.append({
                "Author": author,
                "Number of Articles": len(citations),
                "Total Citations": sum(citations),
                "Average Citations": sum(citations) / len(citations) if citations else 0,
                "h-index": h_index(citations),
                "g-index": g_index(citations)
            })

        df_results = pd.DataFrame(results)

        # --- Main Results Table ---
        st.subheader("Author Metrics Table")
        st.dataframe(df_results)
        st.download_button("Download Author Metrics as CSV",
                           df_results.to_csv(index=False).encode("utf-8"),
                           "author_metrics.csv",
                           "text/csv")

        # --- Top 10 Rankings ---
        st.subheader("Top 10 Authors by Metric")

        col1, col2 = st.columns(2)

        # --- Total Citations ---
        with col1:
            st.markdown("**Top 10 by Total Citations**")
            top_total = df_results.sort_values("Total Citations", ascending=False).head(10)
            st.dataframe(top_total)
            st.download_button("Download CSV",
                               top_total.to_csv(index=False).encode("utf-8"),
                               "top10_total_citations.csv",
                               "text/csv")

            # --- Average Citations ---
            st.markdown("**Top 10 by Average Citations**")
            top_avg = df_results.sort_values("Average Citations", ascending=False).head(10)
            st.dataframe(top_avg)
            st.download_button("Download CSV",
                               top_avg.to_csv(index=False).encode("utf-8"),
                               "top10_avg_citations.csv",
                               "text/csv")

        # --- Number of Articles and h-index ---
        with col2:
            st.markdown("**Top 10 by Number of Articles**")
            top_articles = df_results.sort_values("Number of Articles", ascending=False).head(10)
            st.dataframe(top_articles)
            st.download_button("Download CSV",
                               top_articles.to_csv(index=False).encode("utf-8"),
                               "top10_articles.csv",
                               "text/csv")

            st.markdown("**Top 10 by h-index**")
            top_h = df_results.sort_values("h-index", ascending=False).head(10)
            st.dataframe(top_h)
            st.download_button("Download CSV",
                               top_h.to_csv(index=False).encode("utf-8"),
                               "top10_h_index.csv",
                               "text/csv")

        # --- g-index (separately displayed) ---
        st.markdown("**Top 10 by g-index**")
        top_g = df_results.sort_values("g-index", ascending=False).head(10)
        st.dataframe(top_g)
        st.download_button("Download CSV",
                           top_g.to_csv(index=False).encode("utf-8"),
                           "top10_g_index.csv",
                           "text/csv")

        # --- Missing Citation Information ---
        st.subheader("Articles Missing Citation Information")

        missing_citations = df[df['Times Cited'].isna()][['Title', 'Publication year']]
        count_missing = len(missing_citations)
        total_articles = len(df)
        percentage_missing = (count_missing / total_articles * 100) if total_articles > 0 else 0

        st.markdown(f"**{count_missing} articles** are missing citation information "
                    f"({percentage_missing:.2f}% of total {total_articles}).")

        if count_missing > 0:
            st.dataframe(missing_citations)
            st.download_button("Download Missing Citations as CSV",
                               missing_citations.to_csv(index=False).encode("utf-8"),
                               "missing_citations.csv",
                               "text/csv")

       # --- Pie Chart of Missing Citations by Year ---
        st.subheader("Distribution of Missing Citation Information by Year")

        year_counts = missing_citations['Publication year'].value_counts()

        # Sort years by percentage (descending order)
        year_counts = year_counts.sort_values(ascending=False)

        fig, ax = plt.subplots()
        ax.pie(year_counts, labels=year_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio makes the pie a circle
        st.pyplot(fig)

        # --- Table of Missing Citations by Year ---
        st.subheader("Table of Missing Citation Information by Year")

        total_missing = len(missing_citations)

        if total_missing == 0:
            st.info("No articles with missing citation information to summarize.")
        else:
            # Count per year (keep 'Unknown' if year is missing)
            year_counts = (
                missing_citations
                .assign(**{'Publication year': missing_citations['Publication year'].fillna('Unknown')})
                .groupby('Publication year')
                .size()
                .sort_values(ascending=False)  # sort by count/percentage desc
            )

            year_table = year_counts.reset_index(name="Count")
            year_table["Percentage"] = (year_table["Count"] / total_missing * 100).round(2)

            st.dataframe(year_table)
            st.download_button(
                "Download Missing-by-Year (CSV)",
                year_table.to_csv(index=False).encode("utf-8"),
                "missing_citations_by_year.csv",
                "text/csv"
            )

                    # --- Most Cited Articles per Year (with >200 citations) ---
        st.subheader("Most Cited Articles per Year (Citations > 100)")

        # Filter articles with more than 200 citations
        high_cited = df[df["Times Cited"] > 100][["Title", "Publication year", "Times Cited"]]

        if high_cited.empty:
            st.info("No articles with more than 200 citations found.")
        else:
            # Sort by year and citations
            high_cited = high_cited.sort_values(["Publication year", "Times Cited"], ascending=[True, False])

            # Plot: bar chart with year on x-axis, citations on y-axis
            fig, ax = plt.subplots(figsize=(10, 6))
            for year, group in high_cited.groupby("Publication year"):
                ax.bar(group["Title"], group["Times Cited"], label=year)

            ax.set_xlabel("Article Title")
            ax.set_ylabel("Times Cited")
            ax.set_title("Most Cited Articles per Year (Citations > 200)")
            ax.legend(title="Publication Year")
            plt.xticks(rotation=90)

            st.pyplot(fig)

            # Show data table
            st.dataframe(high_cited)

            st.download_button(
                "Download Most Cited Articles per Year (CSV)",
                high_cited.to_csv(index=False).encode("utf-8"),
                "most_cited_articles_per_year.csv",
                "text/csv"
            )

                    # --- Authors with More Than 100 Total Citations ---
        st.subheader("Authors with More Than 100 Total Citations")

        authors_over_100 = df_results[df_results["Total Citations"] > 100] \
                              .sort_values("Total Citations", ascending=False)

        if authors_over_100.empty:
            st.info("No authors with more than 100 citations found.")
        else:
            st.dataframe(authors_over_100)

            st.download_button(
                "Download Authors >100 Citations (CSV)",
                authors_over_100.to_csv(index=False).encode("utf-8"),
                "authors_over_100_citations.csv",
                "text/csv"
            )

            # --- Number of Articles per Year ---
        st.subheader("Number of Articles per Year")

        if "Publication year" not in df.columns:
            st.warning("No 'Publication year' column found in the dataset.")
        else:
            # Count articles per year
            articles_per_year = (
                df["Publication year"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            articles_per_year.plot(kind="bar", ax=ax)

            ax.set_xlabel("Publication Year")
            ax.set_ylabel("Number of Articles")
            ax.set_title("Number of Articles per Year")

            st.pyplot(fig)

            # Show data table
            year_table = articles_per_year.reset_index()
            year_table.columns = ["Publication Year", "Number of Articles"]
            st.dataframe(year_table)

            st.download_button(
                "Download Articles per Year (CSV)",
                year_table.to_csv(index=False).encode("utf-8"),
                "articles_per_year.csv",
                "text/csv"
            )

            # --- Visualization of h-index and g-index distributions ---
        st.subheader("Distribution of h-index and g-index Across Authors")

        # Histogram of h-index
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_results["h-index"], bins=range(0, df_results["h-index"].max() + 2), edgecolor="black")
        ax.set_xlabel("h-index")
        ax.set_ylabel("Number of Authors")
        ax.set_title("Distribution of h-index Across Authors")
        st.pyplot(fig)

        # Histogram of g-index
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_results["g-index"], bins=range(0, df_results["g-index"].max() + 2), edgecolor="black")
        ax.set_xlabel("g-index")
        ax.set_ylabel("Number of Authors")
        ax.set_title("Distribution of g-index Across Authors")
        st.pyplot(fig)

        # Scatter plot: h-index vs g-index
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df_results["h-index"], df_results["g-index"], alpha=0.7)
        ax.set_xlabel("h-index")
        ax.set_ylabel("g-index")
        ax.set_title("h-index vs g-index (per Author)")
        st.pyplot(fig)

                # --- Lorenz Curve of Author Citations with Gini Coefficient ---
        st.subheader("Lorenz Curve of Citations Across Authors")

        # Sort authors by total citations
        sorted_citations = np.sort(df_results["Total Citations"].values)
        cumulative_citations = np.cumsum(sorted_citations)
        cumulative_citations = cumulative_citations / cumulative_citations[-1]  # normalize to 1
        x_axis = np.arange(1, len(sorted_citations) + 1) / len(sorted_citations)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_axis, cumulative_citations, label="Lorenz Curve", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Equality Line")
        ax.set_xlabel("Cumulative Share of Authors")
        ax.set_ylabel("Cumulative Share of Citations")
        ax.set_title("Lorenz Curve of Citations")

        # Compute Gini coefficient
        n = len(sorted_citations)
        cumulative_sum = np.cumsum(sorted_citations)
        gini = (n + 1 - 2 * np.sum(cumulative_sum) / cumulative_sum[-1]) / n

        st.pyplot(fig)
        st.markdown(f"**Gini Coefficient of Citations:** {gini:.3f}")


        # --- Publications vs Citations per Year ---
        st.subheader("Publications vs Citations per Year")

        if "Publication year" not in df.columns:
            st.warning("No 'Publication year' column found in the dataset.")
        else:
            # Aggregate per year
            df_year = df.copy()
            df_year = df_year.dropna(subset=["Publication year"])
            df_year["Publication year"] = df_year["Publication year"].astype(int)

            pubs_per_year = df_year.groupby("Publication year").size()
            citations_per_year = df_year.groupby("Publication year")["Times Cited"].sum()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(pubs_per_year.index, pubs_per_year.values, marker='o', label="Publications per Year")
            ax.plot(citations_per_year.index, citations_per_year.values, marker='s', label="Citations per Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_title("Publications and Citations per Year")
            ax.legend()
            st.pyplot(fig)

        # --- Average Citations per Paper per Year ---
        st.subheader("Average Citations per Paper per Year")

        # Compute average citations
        avg_citations_per_year = citations_per_year / pubs_per_year

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(avg_citations_per_year.index, avg_citations_per_year.values, marker='o', color='purple')
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Average Citations per Paper")
        ax.set_title("Average Citations per Paper per Year")
        st.pyplot(fig)

        # Show table
        avg_table = avg_citations_per_year.reset_index()
        avg_table.columns = ["Publication Year", "Average Citations per Paper"]
        st.dataframe(avg_table)

        st.download_button(
            "Download Average Citations per Paper (CSV)",
            avg_table.to_csv(index=False).encode("utf-8"),
            "avg_citations_per_paper.csv",
            "text/csv"
        )
