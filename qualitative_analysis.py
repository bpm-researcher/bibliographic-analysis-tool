import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

def show():
    st.title("📊 Model Analysis in Articles")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        # Load the data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Normalize status to string
        df["status"] = df["status"].fillna("").astype(str)

        articles = []
        models = []
        current_article = None

        for _, row in df.iterrows():
            status_norm = str(row.get("status", "")).strip().lower()
            data_val = str(row.get("data", "")).strip()

            if status_norm == "title":
                current_article = data_val
                articles.append({"article": current_article, "models": []})
            elif current_article and data_val:
                # --- normalize "Framework - something" → "Framework"
                if data_val.lower().startswith("framework -"):
                    data_val = "Framework"

                # treat 'yes/sim/y' as used=True; everything else as False
                used_flag = status_norm in {"yes", "sim", "y"}
                models.append({
                    "article": current_article,
                    "model": data_val,
                    "used": bool(used_flag)
                })
                articles[-1]["models"].append(data_val)

        # Ensure columns exist even if models list is empty
        models_df = pd.DataFrame(models, columns=["article", "model", "used"])

        # Cited and used models
        cited = (
            models_df.loc[models_df["used"] == False, "model"]
            .value_counts()
            .rename_axis("Model")
            .reset_index(name="Citations")
        )

        used = (
            models_df.loc[models_df["used"] == True, "model"]
            .value_counts()
            .rename_axis("Model")
            .reset_index(name="Uses")
        )

        # Articles with no models
        articles_without_models = [a["article"] for a in articles if len(a["models"]) == 0]
        articles_without_models_df = pd.DataFrame(articles_without_models, columns=["Articles Without Models"])
        num_articles_without_models = len(articles_without_models)

        # --- Display results ---
        st.subheader("📑 Cited Models")
        st.dataframe(cited)
        st.download_button("Download Cited Models CSV", cited.to_csv(index=False).encode("utf-8"),
                           "cited_models.csv", "text/csv")

        st.subheader("📑 Used Models")
        st.dataframe(used)
        st.download_button("Download Used Models CSV", used.to_csv(index=False).encode("utf-8"),
                           "used_models.csv", "text/csv")

        st.subheader("📑 Articles Without Models")
        st.dataframe(articles_without_models_df)
        st.download_button("Download Articles Without Models CSV",
                           articles_without_models_df.to_csv(index=False).encode("utf-8"),
                           "articles_without_models.csv", "text/csv")

        # --- Bar Chart: Models cited ≥5 (+ "Other") ---
        st.subheader("📊 Cited Models (bar chart - cited ≥5)")
        try:
            if not models_df.empty:
                cited_count = (~models_df["used"]).groupby(models_df["model"]).sum().sort_values(ascending=False)

                top_cited = cited_count[cited_count >= 5]
                other_cited = cited_count[cited_count < 5].sum()

                if other_cited > 0:
                    top_cited["Other models"] = other_cited

                if num_articles_without_models > 0:
                    top_cited["Articles without models"] = num_articles_without_models

                fig_c, ax_c = plt.subplots(figsize=(8, 5))
                bars_c = ax_c.bar(top_cited.index, top_cited.values)

                for bar in bars_c:
                    yval = bar.get_height()
                    ax_c.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval),
                              ha='center', va='bottom')

                ax_c.set_ylabel("Citation Count")
                ax_c.set_xlabel("Model")
                ax_c.set_title("Cited Models (grouping <5 as 'Other')")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig_c)
        except Exception:
            st.error("not valid csv text")

        # --- Cited Models per Year ---
        st.subheader("📈 Cited Models by Year")
        years = []
        for a in articles:
            match = re.search(r"(19|20)\d{2}", a["article"])
            if match:
                year = match.group(0)
                for m in a["models"]:
                    years.append(year)

        if years:
            years_df = pd.Series(years).value_counts().sort_index()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bars2 = ax2.bar(years_df.index, years_df.values)

            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval),
                         ha='center', va='bottom')

            ax2.set_ylabel("Number of Cited Models")
            ax2.set_xlabel("Year")
            ax2.set_title("Cited Models by Year")
            st.pyplot(fig2)
        else:
            st.info("Could not extract years from article titles.")

        # --- Summary table: citations and uses per model ---
        summary = models_df.groupby("model").agg(
            citations=("used", lambda x: (~x).sum()),
            uses=("used", "sum")
        ).reset_index()

        st.subheader("📋 Summary Table: Citations and Uses")
        st.dataframe(summary)
        st.download_button("Download Summary Table CSV", summary.to_csv(index=False).encode("utf-8"), "summary_table.csv", "text/csv")

        # --- Line Chart: Top 10 Cited Models ---
        top_models = summary[summary["citations"] >= 5]

        plt.figure(figsize=(10, 6))
        plt.plot(top_models["model"], top_models["citations"], marker="o", label="Citations")
        plt.plot(top_models["model"], top_models["uses"], marker="o", label="Uses")
        plt.title("Models with ≥5 Citations: Citations vs Uses")
        plt.xlabel("Model")
        plt.ylabel("Quantity")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        st.pyplot(plt)

        # --- Specific model lists ---
        only_used = summary[(summary["citations"] == 0) & (summary["uses"] > 0)]
        st.subheader("📋 Models Only Used (Not Cited)")
        if not only_used.empty:
            st.dataframe(only_used)
            st.download_button("Download Only Used Models CSV", only_used.to_csv(index=False).encode("utf-8"), "only_used_models.csv", "text/csv")
        else:
            st.info("No models are used but not cited.")

        only_cited = summary[(summary["citations"] > 0) & (summary["uses"] == 0)]
        st.subheader("📋 Models Only Cited (Not Used)")
        if not only_cited.empty:
            st.dataframe(only_cited)
            st.download_button("Download Only Cited Models CSV", only_cited.to_csv(index=False).encode("utf-8"), "only_cited_models.csv", "text/csv")
        else:
            st.info("No models are cited but not used.")

        used_more_than_cited = summary[summary["uses"] > summary["citations"]]
        st.subheader("📋 Models Used More Than Cited")
        if not used_more_than_cited.empty:
            st.dataframe(used_more_than_cited)
            st.download_button("Download Models Used More Than Cited CSV", used_more_than_cited.to_csv(index=False).encode("utf-8"), "used_more_than_cited.csv", "text/csv")
        else:
            st.info("No models are used more than cited.")

        cited_more_than_used = summary[summary["citations"] > summary["uses"]]
        st.subheader("📋 Models Cited More Than Used")
        if not cited_more_than_used.empty:
            st.dataframe(cited_more_than_used)
            st.download_button("Download Models Cited More Than Used CSV", cited_more_than_used.to_csv(index=False).encode("utf-8"), "cited_more_than_used.csv", "text/csv")
        else:
            st.info("No models are cited more than used.")
